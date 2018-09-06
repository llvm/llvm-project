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
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Chrono.h"
#include "llvm/Support/Errno.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SourceMgr.h"
#include <istream>

using namespace llvm;
using namespace clang;
using namespace clangd;

namespace {
static Key<json::Value> RequestID;
static Key<JSONOutput *> RequestOut;

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

void JSONOutput::writeMessage(const json::Value &Message) {
  std::string S;
  llvm::raw_string_ostream OS(S);
  if (Pretty)
    OS << llvm::formatv("{0:2}", Message);
  else
    OS << Message;
  OS.flush();

  {
    std::lock_guard<std::mutex> Guard(StreamMutex);
    Outs << "Content-Length: " << S.size() << "\r\n\r\n" << S;
    Outs.flush();
  }
  vlog(">>> {0}\n", S);
}

void JSONOutput::log(Logger::Level Level,
                     const llvm::formatv_object_base &Message) {
  if (Level < MinLevel)
    return;
  llvm::sys::TimePoint<> Timestamp = std::chrono::system_clock::now();
  trace::log(Message);
  std::lock_guard<std::mutex> Guard(StreamMutex);
  Logs << llvm::formatv("{0}[{1:%H:%M:%S.%L}] {2}\n", indicator(Level),
                        Timestamp, Message);
  Logs.flush();
}

void JSONOutput::mirrorInput(const Twine &Message) {
  if (!InputMirror)
    return;

  *InputMirror << Message;
  InputMirror->flush();
}

void clangd::reply(json::Value &&Result) {
  auto ID = getRequestId();
  if (!ID) {
    elog("Attempted to reply to a notification!");
    return;
  }
  RequestSpan::attach([&](json::Object &Args) { Args["Reply"] = Result; });
  log("--> reply({0})", *ID);
  Context::current()
      .getExisting(RequestOut)
      ->writeMessage(json::Object{
          {"jsonrpc", "2.0"},
          {"id", *ID},
          {"result", std::move(Result)},
      });
}

void clangd::replyError(ErrorCode Code, const llvm::StringRef &Message) {
  elog("Error {0}: {1}", static_cast<int>(Code), Message);
  RequestSpan::attach([&](json::Object &Args) {
    Args["Error"] = json::Object{{"code", static_cast<int>(Code)},
                                 {"message", Message.str()}};
  });

  if (auto ID = getRequestId()) {
    log("--> reply({0}) error: {1}", *ID, Message);
    Context::current()
        .getExisting(RequestOut)
        ->writeMessage(json::Object{
            {"jsonrpc", "2.0"},
            {"id", *ID},
            {"error", json::Object{{"code", static_cast<int>(Code)},
                                   {"message", Message}}},
        });
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
      .getExisting(RequestOut)
      ->writeMessage(json::Object{
          {"jsonrpc", "2.0"},
          {"id", ID},
          {"method", Method},
          {"params", std::move(Params)},
      });
}

void JSONRPCDispatcher::registerHandler(StringRef Method, Handler H) {
  assert(!Handlers.count(Method) && "Handler already registered!");
  Handlers[Method] = std::move(H);
}

static void logIncomingMessage(const llvm::Optional<json::Value> &ID,
                               llvm::Optional<StringRef> Method,
                               const json::Object *Error) {
  if (Method) { // incoming request
    if (ID)     // call
      log("<-- {0}({1})", *Method, *ID);
    else // notification
      log("<-- {0}", *Method);
  } else if (ID) { // response, ID must be provided
    if (Error)
      log("<-- reply({0}) error: {1}", *ID,
          Error->getString("message").getValueOr("<no message>"));
    else
      log("<-- reply({0})", *ID);
  }
}

bool JSONRPCDispatcher::call(const json::Value &Message,
                             JSONOutput &Out) const {
  // Message must be an object with "jsonrpc":"2.0".
  auto *Object = Message.getAsObject();
  if (!Object || Object->getString("jsonrpc") != Optional<StringRef>("2.0"))
    return false;
  // ID may be any JSON value. If absent, this is a notification.
  llvm::Optional<json::Value> ID;
  if (auto *I = Object->get("id"))
    ID = std::move(*I);
  auto Method = Object->getString("method");
  logIncomingMessage(ID, Method, Object->getObject("error"));
  if (!Method) // We only handle incoming requests, and ignore responses.
    return false;
  // Params should be given, use null if not.
  json::Value Params = nullptr;
  if (auto *P = Object->get("params"))
    Params = std::move(*P);

  auto I = Handlers.find(*Method);
  auto &Handler = I != Handlers.end() ? I->second : UnknownHandler;

  // Create a Context that contains request information.
  WithContextValue WithRequestOut(RequestOut, &Out);
  llvm::Optional<WithContextValue> WithID;
  if (ID)
    WithID.emplace(RequestID, *ID);

  // Create a tracing Span covering the whole request lifetime.
  trace::Span Tracer(*Method);
  if (ID)
    SPAN_ATTACH(Tracer, "ID", *ID);
  SPAN_ATTACH(Tracer, "Params", Params);

  // Stash a reference to the span args, so later calls can add metadata.
  WithContext WithRequestSpan(RequestSpan::stash(Tracer));
  Handler(std::move(Params));
  return true;
}

// Tries to read a line up to and including \n.
// If failing, feof() or ferror() will be set.
static bool readLine(std::FILE *In, std::string &Out) {
  static constexpr int BufSize = 1024;
  size_t Size = 0;
  Out.clear();
  for (;;) {
    Out.resize(Size + BufSize);
    // Handle EINTR which is sent when a debugger attaches on some platforms.
    if (!llvm::sys::RetryAfterSignal(nullptr, ::fgets, &Out[Size], BufSize, In))
      return false;
    clearerr(In);
    // If the line contained null bytes, anything after it (including \n) will
    // be ignored. Fortunately this is not a legal header or JSON.
    size_t Read = std::strlen(&Out[Size]);
    if (Read > 0 && Out[Size + Read - 1] == '\n') {
      Out.resize(Size + Read);
      return true;
    }
    Size += Read;
  }
}

// Returns None when:
//  - ferror() or feof() are set.
//  - Content-Length is missing or empty (protocol error)
static llvm::Optional<std::string> readStandardMessage(std::FILE *In,
                                                       JSONOutput &Out) {
  // A Language Server Protocol message starts with a set of HTTP headers,
  // delimited  by \r\n, and terminated by an empty line (\r\n).
  unsigned long long ContentLength = 0;
  std::string Line;
  while (true) {
    if (feof(In) || ferror(In) || !readLine(In, Line))
      return llvm::None;

    Out.mirrorInput(Line);
    llvm::StringRef LineRef(Line);

    // We allow comments in headers. Technically this isn't part
    // of the LSP specification, but makes writing tests easier.
    if (LineRef.startswith("#"))
      continue;

    // Content-Length is a mandatory header, and the only one we handle.
    if (LineRef.consume_front("Content-Length: ")) {
      if (ContentLength != 0) {
        elog("Warning: Duplicate Content-Length header received. "
             "The previous value for this message ({0}) was ignored.",
             ContentLength);
      }
      llvm::getAsUnsignedInteger(LineRef.trim(), 0, ContentLength);
      continue;
    } else if (!LineRef.trim().empty()) {
      // It's another header, ignore it.
      continue;
    } else {
      // An empty line indicates the end of headers.
      // Go ahead and read the JSON.
      break;
    }
  }

  // The fuzzer likes crashing us by sending "Content-Length: 9999999999999999"
  if (ContentLength > 1 << 30) { // 1024M
    elog("Refusing to read message with long Content-Length: {0}. "
         "Expect protocol errors",
         ContentLength);
    return llvm::None;
  }
  if (ContentLength == 0) {
    log("Warning: Missing Content-Length header, or zero-length message.");
    return llvm::None;
  }

  std::string JSON(ContentLength, '\0');
  for (size_t Pos = 0, Read; Pos < ContentLength; Pos += Read) {
    // Handle EINTR which is sent when a debugger attaches on some platforms.
    Read = llvm::sys::RetryAfterSignal(0u, ::fread, &JSON[Pos], 1,
                                       ContentLength - Pos, In);
    Out.mirrorInput(StringRef(&JSON[Pos], Read));
    if (Read == 0) {
      elog("Input was aborted. Read only {0} bytes of expected {1}.", Pos,
           ContentLength);
      return llvm::None;
    }
    clearerr(In); // If we're done, the error was transient. If we're not done,
                  // either it was transient or we'll see it again on retry.
    Pos += Read;
  }
  return std::move(JSON);
}

// For lit tests we support a simplified syntax:
// - messages are delimited by '---' on a line by itself
// - lines starting with # are ignored.
// This is a testing path, so favor simplicity over performance here.
// When returning None, feof() or ferror() will be set.
static llvm::Optional<std::string> readDelimitedMessage(std::FILE *In,
                                                        JSONOutput &Out) {
  std::string JSON;
  std::string Line;
  while (readLine(In, Line)) {
    auto LineRef = llvm::StringRef(Line).trim();
    if (LineRef.startswith("#")) // comment
      continue;

    // found a delimiter
    if (LineRef.rtrim() == "---")
      break;

    JSON += Line;
  }

  if (ferror(In)) {
    elog("Input error while reading message!");
    return llvm::None;
  } else { // Including EOF
    Out.mirrorInput(
        llvm::formatv("Content-Length: {0}\r\n\r\n{1}", JSON.size(), JSON));
    return std::move(JSON);
  }
}

// The use of C-style std::FILE* IO deserves some explanation.
// Previously, std::istream was used. When a debugger attached on MacOS, the
// process received EINTR, the stream went bad, and clangd exited.
// A retry-on-EINTR loop around reads solved this problem, but caused clangd to
// sometimes hang rather than exit on other OSes. The interaction between
// istreams and signals isn't well-specified, so it's hard to get this right.
// The C APIs seem to be clearer in this respect.
void clangd::runLanguageServerLoop(std::FILE *In, JSONOutput &Out,
                                   JSONStreamStyle InputStyle,
                                   JSONRPCDispatcher &Dispatcher,
                                   bool &IsDone) {
  auto &ReadMessage =
      (InputStyle == Delimited) ? readDelimitedMessage : readStandardMessage;
  while (!IsDone && !feof(In)) {
    if (ferror(In)) {
      elog("IO error: {0}", llvm::sys::StrError());
      return;
    }
    if (auto JSON = ReadMessage(In, Out)) {
      if (auto Doc = json::parse(*JSON)) {
        // Log the formatted message.
        vlog(Out.Pretty ? "<<< {0:2}\n" : "<<< {0}\n", *Doc);
        // Finally, execute the action for this JSON message.
        if (!Dispatcher.call(*Doc, Out))
          elog("JSON dispatch failed!");
      } else {
        // Parse error. Log the raw message.
        vlog("<<< {0}\n", *JSON);
        elog("JSON parse error: {0}", llvm::toString(Doc.takeError()));
      }
    }
  }
}

const json::Value *clangd::getRequestId() {
  return Context::current().get(RequestID);
}
