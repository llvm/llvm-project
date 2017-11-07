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
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/YAMLParser.h"
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
  Logs << Message;
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
    Out.log("Attempted to reply to a notification!\n");
    return;
  }
  Out.writeMessage(json::obj{
      {"jsonrpc", "2.0"},
      {"id", *ID},
      {"result", std::move(Result)},
  });
}

void RequestContext::replyError(ErrorCode code, const llvm::StringRef &Message) {
  Out.log("Error " + Twine(static_cast<int>(code)) + ": " + Message + "\n");
  if (ID) {
    Out.writeMessage(json::obj{
        {"jsonrpc", "2.0"},
        {"id", *ID},
        {"error", json::obj{{"code", static_cast<int>(code)}, {"message", Message}}},
    });
  }
}

void RequestContext::call(StringRef Method, json::Expr &&Params) {
  // FIXME: Generate/Increment IDs for every request so that we can get proper
  // replies once we need to.
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

static void
callHandler(const llvm::StringMap<JSONRPCDispatcher::Handler> &Handlers,
            llvm::yaml::ScalarNode *Method, llvm::Optional<json::Expr> ID,
            llvm::yaml::MappingNode *Params,
            const JSONRPCDispatcher::Handler &UnknownHandler, JSONOutput &Out) {
  llvm::SmallString<64> MethodStorage;
  llvm::StringRef MethodStr = Method->getValue(MethodStorage);
  auto I = Handlers.find(MethodStr);
  auto &Handler = I != Handlers.end() ? I->second : UnknownHandler;
  trace::Span Tracer(MethodStr);
  Handler(RequestContext(Out, std::move(ID)), Params);
}

bool JSONRPCDispatcher::call(StringRef Content, JSONOutput &Out) const {
  llvm::SourceMgr SM;
  llvm::yaml::Stream YAMLStream(Content, SM);

  auto Doc = YAMLStream.begin();
  if (Doc == YAMLStream.end())
    return false;

  auto *Object = dyn_cast_or_null<llvm::yaml::MappingNode>(Doc->getRoot());
  if (!Object)
    return false;

  llvm::yaml::ScalarNode *Version = nullptr;
  llvm::yaml::ScalarNode *Method = nullptr;
  llvm::yaml::MappingNode *Params = nullptr;
  llvm::Optional<json::Expr> ID;
  for (auto &NextKeyValue : *Object) {
    auto *KeyString =
        dyn_cast_or_null<llvm::yaml::ScalarNode>(NextKeyValue.getKey());
    if (!KeyString)
      return false;

    llvm::SmallString<10> KeyStorage;
    StringRef KeyValue = KeyString->getValue(KeyStorage);
    llvm::yaml::Node *Value = NextKeyValue.getValue();
    if (!Value)
      return false;

    if (KeyValue == "jsonrpc") {
      // This should be "2.0". Always.
      Version = dyn_cast<llvm::yaml::ScalarNode>(Value);
      if (!Version || Version->getRawValue() != "\"2.0\"")
        return false;
    } else if (KeyValue == "method") {
      Method = dyn_cast<llvm::yaml::ScalarNode>(Value);
    } else if (KeyValue == "id") {
      // ID may be either a string or a number.
      if (auto *IdNode = dyn_cast<llvm::yaml::ScalarNode>(Value)) {
        llvm::SmallString<32> S;
        llvm::StringRef V = IdNode->getValue(S);
        if (IdNode->getRawValue().startswith("\"")) {
          ID.emplace(V.str());
        } else {
          double D;
          if (!V.getAsDouble(D))
            ID.emplace(D);
        }
      }
    } else if (KeyValue == "params") {
      if (!Method)
        return false;
      // We have to interleave the call of the function here, otherwise the
      // YAMLParser will die because it can't go backwards. This is unfortunate
      // because it will break clients that put the id after params. A possible
      // fix would be to split the parsing and execution phases.
      Params = dyn_cast<llvm::yaml::MappingNode>(Value);
      callHandler(Handlers, Method, std::move(ID), Params, UnknownHandler, Out);
      return true;
    } else {
      return false;
    }
  }

  // In case there was a request with no params, call the handler on the
  // leftovers.
  if (!Method)
    return false;
  callHandler(Handlers, Method, std::move(ID), nullptr, UnknownHandler, Out);

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

      // We allow YAML-style comments in headers. Technically this isn't part
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
                  std::to_string(ContentLength) + ") was ignored.\n");
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
              " bytes.\n");
      continue;
    }

    if (ContentLength > 0) {
      std::vector<char> JSON(ContentLength + 1, '\0');
      llvm::StringRef JSONRef;
      {
        trace::Span Tracer("Reading request");
        // Now read the JSON. Insert a trailing null byte as required by the
        // YAML parser.
        In.read(JSON.data(), ContentLength);
        Out.mirrorInput(StringRef(JSON.data(), In.gcount()));

        // If the stream is aborted before we read ContentLength bytes, In
        // will have eofbit and failbit set.
        if (!In) {
          Out.log("Input was aborted. Read only " +
                  std::to_string(In.gcount()) + " bytes of expected " +
                  std::to_string(ContentLength) + ".\n");
          break;
        }

        JSONRef = StringRef(JSON.data(), ContentLength);
      }

      // Log the message.
      Out.log("<-- " + JSONRef + "\n");

      // Finally, execute the action for this JSON message.
      if (!Dispatcher.call(JSONRef, Out))
        Out.log("JSON dispatch failed!\n");

      // If we're done, exit the loop.
      if (IsDone)
        break;
    } else {
      Out.log("Warning: Missing Content-Length header, or message has zero "
              "length.\n");
    }
  }
}
