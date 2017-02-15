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
