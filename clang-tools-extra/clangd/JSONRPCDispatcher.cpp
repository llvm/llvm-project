//===--- JSONRPCDispatcher.cpp - Main JSON parser entry point -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "JSONRPCDispatcher.h"
#include "JSONExpr.h"
#include "ProtocolHandlers.h"
#include "Trace.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Chrono.h"
#include "llvm/Support/SourceMgr.h"
#include <istream>

using namespace clang;
using namespace clangd;

namespace {
static Key<json::Expr> RequestID;
static Key<JSONOutput *> RequestOut;

// When tracing, we trace a request and attach the repsonse in reply().
// Because the Span isn't available, we find the current request using Context.
class RequestSpan {
  RequestSpan(json::obj *Args) : Args(Args) {}
  std::mutex Mu;
  json::obj *Args;
  static Key<std::unique_ptr<RequestSpan>> RSKey;

public:
  // Return a context that's aware of the enclosing request, identified by Span.
  static Context stash(const trace::Span &Span) {
    return Context::current().derive(
        RSKey, std::unique_ptr<RequestSpan>(new RequestSpan(Span.Args)));
  }

  // If there's an enclosing request and the tracer is interested, calls \p F
  // with a json::obj where request info can be added.
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

void JSONOutput::writeMessage(const json::Expr &Message) {
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
  log(llvm::Twine("--> ") + S);
}

void JSONOutput::log(const Twine &Message) {
  llvm::sys::TimePoint<> Timestamp = std::chrono::system_clock::now();
  trace::log(Message);
  std::lock_guard<std::mutex> Guard(StreamMutex);
  Logs << llvm::formatv("[{0:%H:%M:%S.%L}] {1}\n", Timestamp, Message);
  Logs.flush();
}

void JSONOutput::mirrorInput(const Twine &Message) {
  if (!InputMirror)
    return;

  *InputMirror << Message;
  InputMirror->flush();
}

void clangd::reply(json::Expr &&Result) {
  auto ID = Context::current().get(RequestID);
  if (!ID) {
    log("Attempted to reply to a notification!");
    return;
  }
  RequestSpan::attach([&](json::obj &Args) { Args["Reply"] = Result; });
  Context::current()
      .getExisting(RequestOut)
      ->writeMessage(json::obj{
          {"jsonrpc", "2.0"},
          {"id", *ID},
          {"result", std::move(Result)},
      });
}

void clangd::replyError(ErrorCode code, const llvm::StringRef &Message) {
  log("Error " + Twine(static_cast<int>(code)) + ": " + Message);
  RequestSpan::attach([&](json::obj &Args) {
    Args["Error"] =
        json::obj{{"code", static_cast<int>(code)}, {"message", Message.str()}};
  });

  if (auto ID = Context::current().get(RequestID)) {
    Context::current()
        .getExisting(RequestOut)
        ->writeMessage(json::obj{
            {"jsonrpc", "2.0"},
            {"id", *ID},
            {"error",
             json::obj{{"code", static_cast<int>(code)}, {"message", Message}}},
        });
  }
}

void clangd::call(StringRef Method, json::Expr &&Params) {
  // FIXME: Generate/Increment IDs for every request so that we can get proper
  // replies once we need to.
  RequestSpan::attach([&](json::obj &Args) {
    Args["Call"] = json::obj{{"method", Method.str()}, {"params", Params}};
  });
  Context::current()
      .getExisting(RequestOut)
      ->writeMessage(json::obj{
          {"jsonrpc", "2.0"},
          {"id", 1},
          {"method", Method},
          {"params", std::move(Params)},
      });
}

void JSONRPCDispatcher::registerHandler(StringRef Method, Handler H) {
  assert(!Handlers.count(Method) && "Handler already registered!");
  Handlers[Method] = std::move(H);
}

bool JSONRPCDispatcher::call(const json::Expr &Message, JSONOutput &Out) const {
  // Message must be an object with "jsonrpc":"2.0".
  auto *Object = Message.asObject();
  if (!Object || Object->getString("jsonrpc") != Optional<StringRef>("2.0"))
    return false;
  // ID may be any JSON value. If absent, this is a notification.
  llvm::Optional<json::Expr> ID;
  if (auto *I = Object->get("id"))
    ID = std::move(*I);
  // Method must be given.
  auto Method = Object->getString("method");
  if (!Method)
    return false;
  // Params should be given, use null if not.
  json::Expr Params = nullptr;
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

static llvm::Optional<std::string> readStandardMessage(std::istream &In,
                                                       JSONOutput &Out) {
  // A Language Server Protocol message starts with a set of HTTP headers,
  // delimited  by \r\n, and terminated by an empty line (\r\n).
  unsigned long long ContentLength = 0;
  while (In.good()) {
    std::string Line;
    std::getline(In, Line);
    if (!In.good() && errno == EINTR) {
      In.clear();
      continue;
    }

    Out.mirrorInput(Line);
    // Mirror '\n' that gets consumed by std::getline, but is not included in
    // the resulting Line.
    // Note that '\r' is part of Line, so we don't need to mirror it
    // separately.
    if (!In.eof())
      Out.mirrorInput("\n");

    llvm::StringRef LineRef(Line);

    // We allow comments in headers. Technically this isn't part
    // of the LSP specification, but makes writing tests easier.
    if (LineRef.startswith("#"))
      continue;

    // Content-Type is a specified header, but does nothing.
    // Content-Length is a mandatory header. It specifies the length of the
    // following JSON.
    // It is unspecified what sequence headers must be supplied in, so we
    // allow any sequence.
    // The end of headers is signified by an empty line.
    if (LineRef.consume_front("Content-Length: ")) {
      if (ContentLength != 0) {
        log("Warning: Duplicate Content-Length header received. "
            "The previous value for this message (" +
            llvm::Twine(ContentLength) + ") was ignored.\n");
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

  // Guard against large messages. This is usually a bug in the client code
  // and we don't want to crash downstream because of it.
  if (ContentLength > 1 << 30) { // 1024M
    In.ignore(ContentLength);
    log("Skipped overly large message of " + Twine(ContentLength) +
        " bytes.\n");
    return llvm::None;
  }

  if (ContentLength > 0) {
    std::string JSON(ContentLength, '\0');
    In.read(&JSON[0], ContentLength);
    Out.mirrorInput(StringRef(JSON.data(), In.gcount()));

    // If the stream is aborted before we read ContentLength bytes, In
    // will have eofbit and failbit set.
    if (!In) {
      log("Input was aborted. Read only " + llvm::Twine(In.gcount()) +
          " bytes of expected " + llvm::Twine(ContentLength) + ".\n");
      return llvm::None;
    }
    return std::move(JSON);
  } else {
    log("Warning: Missing Content-Length header, or message has zero "
        "length.\n");
    return llvm::None;
  }
}

// For lit tests we support a simplified syntax:
// - messages are delimited by '---' on a line by itself
// - lines starting with # are ignored.
// This is a testing path, so favor simplicity over performance here.
static llvm::Optional<std::string> readDelimitedMessage(std::istream &In,
                                                        JSONOutput &Out) {
  std::string JSON;
  std::string Line;
  while (std::getline(In, Line)) {
    Line.push_back('\n'); // getline() consumed the newline.

    auto LineRef = llvm::StringRef(Line).trim();
    if (LineRef.startswith("#")) // comment
      continue;

    // found a delimiter
    if (LineRef.rtrim() == "---")
      break;

    JSON += Line;
  }

  if (In.bad()) {
    log("Input error while reading message!");
    return llvm::None;
  } else {
    Out.mirrorInput(
        llvm::formatv("Content-Length: {0}\r\n\r\n{1}", JSON.size(), JSON));
    return std::move(JSON);
  }
}

void clangd::runLanguageServerLoop(std::istream &In, JSONOutput &Out,
                                   JSONStreamStyle InputStyle,
                                   JSONRPCDispatcher &Dispatcher,
                                   bool &IsDone) {
  auto &ReadMessage =
      (InputStyle == Delimited) ? readDelimitedMessage : readStandardMessage;
  while (In.good()) {
    if (auto JSON = ReadMessage(In, Out)) {
      if (auto Doc = json::parse(*JSON)) {
        // Log the formatted message.
        log(llvm::formatv(Out.Pretty ? "<-- {0:2}\n" : "<-- {0}\n", *Doc));
        // Finally, execute the action for this JSON message.
        if (!Dispatcher.call(*Doc, Out))
          log("JSON dispatch failed!\n");
      } else {
        // Parse error. Log the raw message.
        log(llvm::formatv("<-- {0}\n" , *JSON));
        log(llvm::Twine("JSON parse error: ") +
            llvm::toString(Doc.takeError()) + "\n");
      }
    }
    // If we're done, exit the loop.
    if (IsDone)
      break;
  }
}
