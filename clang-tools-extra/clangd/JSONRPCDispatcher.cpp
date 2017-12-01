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
#include "llvm/Support/SourceMgr.h"
#include <istream>

using namespace clang;
using namespace clangd;

void JSONOutput::writeMessage(const json::Expr &Message) {
  std::string S;
  llvm::raw_string_ostream OS(S);
  if (Pretty)
    OS << llvm::formatv("{0:2}", Message);
  else
    OS << Message;
  OS.flush();

  std::lock_guard<std::mutex> Guard(StreamMutex);
  // Log without headers.
  Logs << "--> " << S << '\n';
  Logs.flush();

  // Emit message with header.
  Outs << "Content-Length: " << S.size() << "\r\n\r\n" << S;
  Outs.flush();
}

void JSONOutput::log(const Twine &Message) {
  trace::log(Message);
  std::lock_guard<std::mutex> Guard(StreamMutex);
  Logs << Message << '\n';
  Logs.flush();
}

void JSONOutput::mirrorInput(const Twine &Message) {
  if (!InputMirror)
    return;

  *InputMirror << Message;
  InputMirror->flush();
}

void RequestContext::reply(json::Expr &&Result) {
  if (!ID) {
    Out.log("Attempted to reply to a notification!");
    return;
  }
  SPAN_ATTACH(tracer(), "Reply", Result);
  Out.writeMessage(json::obj{
      {"jsonrpc", "2.0"},
      {"id", *ID},
      {"result", std::move(Result)},
  });
}

void RequestContext::replyError(ErrorCode code,
                                const llvm::StringRef &Message) {
  Out.log("Error " + Twine(static_cast<int>(code)) + ": " + Message);
  SPAN_ATTACH(tracer(), "Error",
              (json::obj{{"code", static_cast<int>(code)},
                         {"message", Message.str()}}));
  if (ID) {
    Out.writeMessage(json::obj{
        {"jsonrpc", "2.0"},
        {"id", *ID},
        {"error",
         json::obj{{"code", static_cast<int>(code)}, {"message", Message}}},
    });
  }
}

void RequestContext::call(StringRef Method, json::Expr &&Params) {
  // FIXME: Generate/Increment IDs for every request so that we can get proper
  // replies once we need to.
  SPAN_ATTACH(tracer(), "Call",
              (json::obj{{"method", Method.str()}, {"params", Params}}));
  Out.writeMessage(json::obj{
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
  RequestContext Ctx(Out, *Method, std::move(ID));
  SPAN_ATTACH(Ctx.tracer(), "Params", Params);
  Handler(std::move(Ctx), std::move(Params));
  return true;
}

void clangd::runLanguageServerLoop(std::istream &In, JSONOutput &Out,
                                   JSONRPCDispatcher &Dispatcher,
                                   bool &IsDone) {
  while (In.good()) {
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
          Out.log("Warning: Duplicate Content-Length header received. "
                  "The previous value for this message (" +
                  std::to_string(ContentLength) + ") was ignored.");
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
      Out.log("Skipped overly large message of " + Twine(ContentLength) +
              " bytes.");
      continue;
    }

    if (ContentLength > 0) {
      std::vector<char> JSON(ContentLength);
      llvm::StringRef JSONRef;
      {
        In.read(JSON.data(), ContentLength);
        Out.mirrorInput(StringRef(JSON.data(), In.gcount()));

        // If the stream is aborted before we read ContentLength bytes, In
        // will have eofbit and failbit set.
        if (!In) {
          Out.log("Input was aborted. Read only " +
                  std::to_string(In.gcount()) + " bytes of expected " +
                  std::to_string(ContentLength) + ".");
          break;
        }

        JSONRef = StringRef(JSON.data(), ContentLength);
      }

      if (auto Doc = json::parse(JSONRef)) {
        // Log the formatted message.
        Out.log(llvm::formatv(Out.Pretty ? "<-- {0:2}" : "<-- {0}", *Doc));
        // Finally, execute the action for this JSON message.
        if (!Dispatcher.call(*Doc, Out))
          Out.log("JSON dispatch failed!");
      } else {
        // Parse error. Log the raw message.
        Out.log("<-- " + JSONRef);
        Out.log(llvm::Twine("JSON parse error: ") +
                llvm::toString(Doc.takeError()));
      }

      // If we're done, exit the loop.
      if (IsDone)
        break;
    } else {
      Out.log("Warning: Missing Content-Length header, or message has zero "
              "length.");
    }
  }
}
