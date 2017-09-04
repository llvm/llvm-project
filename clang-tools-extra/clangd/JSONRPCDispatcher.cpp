//===--- JSONRPCDispatcher.cpp - Main JSON parser entry point -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "JSONRPCDispatcher.h"
#include "ProtocolHandlers.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/YAMLParser.h"
#include <istream>

using namespace clang;
using namespace clangd;

void JSONOutput::writeMessage(const Twine &Message) {
  llvm::SmallString<128> Storage;
  StringRef M = Message.toStringRef(Storage);

  std::lock_guard<std::mutex> Guard(StreamMutex);
  // Log without headers.
  Logs << "--> " << M << '\n';
  Logs.flush();

  // Emit message with header.
  Outs << "Content-Length: " << M.size() << "\r\n\r\n" << M;
  Outs.flush();
}

void JSONOutput::log(const Twine &Message) {
  std::lock_guard<std::mutex> Guard(StreamMutex);
  Logs << Message;
  Logs.flush();
}

void Handler::handleMethod(llvm::yaml::MappingNode *Params, StringRef ID) {
  Output.log("Method ignored.\n");
  // Return that this method is unsupported.
  writeMessage(
      R"({"jsonrpc":"2.0","id":)" + ID +
      R"(,"error":{"code":-32601}})");
}

void Handler::handleNotification(llvm::yaml::MappingNode *Params) {
  Output.log("Notification ignored.\n");
}

void JSONRPCDispatcher::registerHandler(StringRef Method,
                                        std::unique_ptr<Handler> H) {
  assert(!Handlers.count(Method) && "Handler already registered!");
  Handlers[Method] = std::move(H);
}

static void
callHandler(const llvm::StringMap<std::unique_ptr<Handler>> &Handlers,
            llvm::yaml::ScalarNode *Method, llvm::yaml::ScalarNode *Id,
            llvm::yaml::MappingNode *Params, Handler *UnknownHandler) {
  llvm::SmallString<10> MethodStorage;
  auto I = Handlers.find(Method->getValue(MethodStorage));
  auto *Handler = I != Handlers.end() ? I->second.get() : UnknownHandler;
  if (Id)
    Handler->handleMethod(Params, Id->getRawValue());
  else
    Handler->handleNotification(Params);
}

bool JSONRPCDispatcher::call(StringRef Content) const {
  llvm::SourceMgr SM;
  llvm::yaml::Stream YAMLStream(Content, SM);

  auto Doc = YAMLStream.begin();
  if (Doc == YAMLStream.end())
    return false;

  auto *Root = Doc->getRoot();
  if (!Root)
    return false;

  auto *Object = dyn_cast<llvm::yaml::MappingNode>(Root);
  if (!Object)
    return false;

  llvm::yaml::ScalarNode *Version = nullptr;
  llvm::yaml::ScalarNode *Method = nullptr;
  llvm::yaml::MappingNode *Params = nullptr;
  llvm::yaml::ScalarNode *Id = nullptr;
  for (auto &NextKeyValue : *Object) {
    auto *KeyString = dyn_cast<llvm::yaml::ScalarNode>(NextKeyValue.getKey());
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
      Id = dyn_cast<llvm::yaml::ScalarNode>(Value);
    } else if (KeyValue == "params") {
      if (!Method)
        return false;
      // We have to interleave the call of the function here, otherwise the
      // YAMLParser will die because it can't go backwards. This is unfortunate
      // because it will break clients that put the id after params. A possible
      // fix would be to split the parsing and execution phases.
      Params = dyn_cast<llvm::yaml::MappingNode>(Value);
      callHandler(Handlers, Method, Id, Params, UnknownHandler.get());
      return true;
    } else {
      return false;
    }
  }

  // In case there was a request with no params, call the handler on the
  // leftovers.
  if (!Method)
    return false;
  callHandler(Handlers, Method, Id, nullptr, UnknownHandler.get());

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
                  "The previous value for this message ("
                  + std::to_string(ContentLength)
                  + ") was ignored.\n");
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

    if (ContentLength > 0) {
      // Now read the JSON. Insert a trailing null byte as required by the YAML
      // parser.
      std::vector<char> JSON(ContentLength + 1, '\0');
      In.read(JSON.data(), ContentLength);

      // If the stream is aborted before we read ContentLength bytes, In
      // will have eofbit and failbit set.
      if (!In) {
        Out.log("Input was aborted. Read only "
                + std::to_string(In.gcount())
                + " bytes of expected "
                + std::to_string(ContentLength)
                + ".\n");
        break;
      }

      llvm::StringRef JSONRef(JSON.data(), ContentLength);
      // Log the message.
      Out.log("<-- " + JSONRef + "\n");

      // Finally, execute the action for this JSON message.
      if (!Dispatcher.call(JSONRef))
        Out.log("JSON dispatch failed!\n");

      // If we're done, exit the loop.
      if (IsDone)
        break;
    } else {
      Out.log( "Warning: Missing Content-Length header, or message has zero "
               "length.\n" );
    }
  }
}
