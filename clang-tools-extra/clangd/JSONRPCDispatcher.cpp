//===--- JSONRPCDispatcher.cpp - Main JSON parser entry point -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "JSONRPCDispatcher.h"
#include "Cancellation.h"
#include "ProtocolHandlers.h"
#include "Trace.h"
#include "Transport.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Errno.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/SourceMgr.h"
#include <istream>

using namespace llvm;
using namespace clang;
using namespace clangd;

namespace {
static Key<json::Value> RequestID;
static Key<Transport *> CurrentTransport;

// When tracing, we trace a request and attach the response in reply().
// Because the Span isn't available, we find the current request using Context.
class RequestSpan {
  RequestSpan(llvm::json::Object *Args) : Args(Args) {}
  std::mutex Mu;
  llvm::json::Object *Args;
  static Key<std::unique_ptr<RequestSpan>> RSKey;

public:
  // Return a context that's aware of the enclosing request, identified by Span.
  static Context stash(const trace::Span &Span) {
    return Context::current().derive(
        RSKey, std::unique_ptr<RequestSpan>(new RequestSpan(Span.Args)));
  }

  // If there's an enclosing request and the tracer is interested, calls \p F
  // with a json::Object where request info can be added.
  template <typename Func> static void attach(Func &&F) {
    auto *RequestArgs = Context::current().get(RSKey);
    if (!RequestArgs || !*RequestArgs || !(*RequestArgs)->Args)
      return;
    std::lock_guard<std::mutex> Lock((*RequestArgs)->Mu);
    F(*(*RequestArgs)->Args);
  }
};
Key<std::unique_ptr<RequestSpan>> RequestSpan::RSKey;
} // namespace

void clangd::reply(json::Value &&Result) {
  auto ID = Context::current().get(RequestID);
  if (!ID) {
    elog("Attempted to reply to a notification!");
    return;
  }
  RequestSpan::attach([&](json::Object &Args) { Args["Reply"] = Result; });
  log("--> reply({0})", *ID);
  Context::current()
      .getExisting(CurrentTransport)
      ->reply(std::move(*ID), std::move(Result));
}

void clangd::replyError(ErrorCode Code, const llvm::StringRef &Message) {
  elog("Error {0}: {1}", static_cast<int>(Code), Message);
  RequestSpan::attach([&](json::Object &Args) {
    Args["Error"] = json::Object{{"code", static_cast<int>(Code)},
                                 {"message", Message.str()}};
  });

  if (auto ID = Context::current().get(RequestID)) {
    log("--> reply({0}) error: {1}", *ID, Message);
    Context::current()
        .getExisting(CurrentTransport)
        ->reply(std::move(*ID), make_error<LSPError>(Message, Code));
  }
}

void clangd::replyError(Error E) {
  handleAllErrors(std::move(E),
                  [](const CancelledError &TCE) {
                    replyError(ErrorCode::RequestCancelled, TCE.message());
                  },
                  [](const ErrorInfoBase &EIB) {
                    replyError(ErrorCode::InvalidParams, EIB.message());
                  });
}

void clangd::call(StringRef Method, json::Value &&Params) {
  RequestSpan::attach([&](json::Object &Args) {
    Args["Call"] = json::Object{{"method", Method.str()}, {"params", Params}};
  });
  // FIXME: Generate/Increment IDs for every request so that we can get proper
  // replies once we need to.
  auto ID = 1;
  log("--> {0}({1})", Method, ID);
  Context::current()
      .getExisting(CurrentTransport)
      ->call(Method, std::move(Params), ID);
}

JSONRPCDispatcher::JSONRPCDispatcher(Handler UnknownHandler)
    : UnknownHandler(std::move(UnknownHandler)) {
  registerHandler("$/cancelRequest", [this](const json::Value &Params) {
    if (auto *O = Params.getAsObject())
      if (auto *ID = O->get("id")) {
        cancelRequest(*ID);
        return true;
      }
    log("Bad cancellation request: {0}", Params);
    return true;
  });
}

void JSONRPCDispatcher::registerHandler(StringRef Method, Handler H) {
  assert(!Handlers.count(Method) && "Handler already registered!");
  Handlers[Method] = std::move(H);
}

bool JSONRPCDispatcher::onCall(StringRef Method, json::Value Params,
                               json::Value ID) {
  log("<-- {0}({1})", Method, ID);
  auto I = Handlers.find(Method);
  auto &Handler = I != Handlers.end() ? I->second : UnknownHandler;

  // Create a Context that contains request information.
  WithContextValue WithID(RequestID, ID);

  // Create a tracing Span covering the whole request lifetime.
  trace::Span Tracer(Method);
  SPAN_ATTACH(Tracer, "ID", ID);
  SPAN_ATTACH(Tracer, "Params", Params);

  // Calls can be canceled by the client. Add cancellation context.
  WithContext WithCancel(cancelableRequestContext(ID));

  // Stash a reference to the span args, so later calls can add metadata.
  WithContext WithRequestSpan(RequestSpan::stash(Tracer));
  return Handler(std::move(Params));
}

bool JSONRPCDispatcher::onNotify(StringRef Method, json::Value Params) {
  log("<-- {0}", Method);
  auto I = Handlers.find(Method);
  auto &Handler = I != Handlers.end() ? I->second : UnknownHandler;

  // Create a tracing Span covering the whole request lifetime.
  trace::Span Tracer(Method);
  SPAN_ATTACH(Tracer, "Params", Params);

  // Stash a reference to the span args, so later calls can add metadata.
  WithContext WithRequestSpan(RequestSpan::stash(Tracer));
  return Handler(std::move(Params));
}

bool JSONRPCDispatcher::onReply(json::Value ID, Expected<json::Value> Result) {
  // We ignore replies, just log them.
  if (Result)
    log("<-- reply({0})", ID);
  else
    log("<-- reply({0}) error: {1}", ID, llvm::toString(Result.takeError()));
  return true;
}

// We run cancelable requests in a context that does two things:
//  - allows cancellation using RequestCancelers[ID]
//  - cleans up the entry in RequestCancelers when it's no longer needed
// If a client reuses an ID, the last one wins and the first cannot be canceled.
Context JSONRPCDispatcher::cancelableRequestContext(const json::Value &ID) {
  auto Task = cancelableTask();
  auto StrID = llvm::to_string(ID);  // JSON-serialize ID for map key.
  auto Cookie = NextRequestCookie++; // No lock, only called on main thread.
  {
    std::lock_guard<std::mutex> Lock(RequestCancelersMutex);
    RequestCancelers[StrID] = {std::move(Task.second), Cookie};
  }
  // When the request ends, we can clean up the entry we just added.
  // The cookie lets us check that it hasn't been overwritten due to ID reuse.
  return Task.first.derive(make_scope_exit([this, StrID, Cookie] {
    std::lock_guard<std::mutex> Lock(RequestCancelersMutex);
    auto It = RequestCancelers.find(StrID);
    if (It != RequestCancelers.end() && It->second.second == Cookie)
      RequestCancelers.erase(It);
  }));
}

void JSONRPCDispatcher::cancelRequest(const json::Value &ID) {
  auto StrID = llvm::to_string(ID);
  std::lock_guard<std::mutex> Lock(RequestCancelersMutex);
  auto It = RequestCancelers.find(StrID);
  if (It != RequestCancelers.end())
    It->second.first(); // Invoke the canceler.
}

llvm::Error JSONRPCDispatcher::runLanguageServerLoop(Transport &Transport) {
  // Propagate transport to all handlers so they can reply.
  WithContextValue WithTransport(CurrentTransport, &Transport);
  return Transport.loop(*this);
}
