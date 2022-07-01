//===--- XPCTransport.cpp - sending and receiving LSP messages over XPC ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Conversion.h"
#include "Protocol.h" // For LSPError
#include "Transport.h"
#include "support/Logger.h"
#include "llvm/Support/Errno.h"

#include <xpc/xpc.h>

using namespace llvm;
using namespace clang;
using namespace clangd;

namespace {

json::Object encodeError(Error E) {
  std::string Message;
  ErrorCode Code = ErrorCode::UnknownErrorCode;
  if (Error Unhandled =
          handleErrors(std::move(E), [&](const LSPError &L) -> Error {
            Message = L.Message;
            Code = L.Code;
            return Error::success();
          }))
    Message = toString(std::move(Unhandled));

  return json::Object{
      {"message", std::move(Message)},
      {"code", int64_t(Code)},
  };
}

Error decodeError(const json::Object &O) {
  std::string Msg =
      std::string(O.getString("message").value_or("Unspecified error"));
  if (auto Code = O.getInteger("code"))
    return make_error<LSPError>(std::move(Msg), ErrorCode(*Code));
  return error("{0}", Msg);
}

// C "closure" for XPCTransport::loop() method
namespace xpcClosure {
void connection_handler(xpc_connection_t clientConnection);
} // namespace xpcClosure

class XPCTransport : public Transport {
public:
  XPCTransport() {}

  void notify(StringRef Method, json::Value Params) override {
    sendMessage(json::Object{
        {"jsonrpc", "2.0"},
        {"method", Method},
        {"params", std::move(Params)},
    });
  }
  void call(StringRef Method, json::Value Params, json::Value ID) override {
    sendMessage(json::Object{
        {"jsonrpc", "2.0"},
        {"id", std::move(ID)},
        {"method", Method},
        {"params", std::move(Params)},
    });
  }
  void reply(json::Value ID, Expected<json::Value> Result) override {
    if (Result) {
      sendMessage(json::Object{
          {"jsonrpc", "2.0"},
          {"id", std::move(ID)},
          {"result", std::move(*Result)},
      });
    } else {
      sendMessage(json::Object{
          {"jsonrpc", "2.0"},
          {"id", std::move(ID)},
          {"error", encodeError(Result.takeError())},
      });
    }
  }

  Error loop(MessageHandler &Handler) override;

private:
  // Needs access to handleMessage() and resetClientConnection()
  friend void xpcClosure::connection_handler(xpc_connection_t clientConnection);

  // Dispatches incoming message to Handler onNotify/onCall/onReply.
  bool handleMessage(json::Value Message, MessageHandler &Handler);
  void sendMessage(json::Value Message) {
    xpc_object_t response = jsonToXpc(Message);
    xpc_connection_send_message(clientConnection, response);
    xpc_release(response);
  }
  void resetClientConnection(xpc_connection_t newClientConnection) {
    clientConnection = newClientConnection;
  }
  xpc_connection_t clientConnection;
};

bool XPCTransport::handleMessage(json::Value Message, MessageHandler &Handler) {
  // Message must be an object with "jsonrpc":"2.0".
  auto *Object = Message.getAsObject();
  if (!Object || Object->getString("jsonrpc") != Optional<StringRef>("2.0")) {
    elog("Not a JSON-RPC 2.0 message: {0:2}", Message);
    return false;
  }
  // ID may be any JSON value. If absent, this is a notification.
  Optional<json::Value> ID;
  if (auto *I = Object->get("id"))
    ID = std::move(*I);
  auto Method = Object->getString("method");
  if (!Method) { // This is a response.
    if (!ID) {
      elog("No method and no response ID: {0:2}", Message);
      return false;
    }
    if (auto *Err = Object->getObject("error"))
      return Handler.onReply(std::move(*ID), decodeError(*Err));
    // Result should be given, use null if not.
    json::Value Result = nullptr;
    if (auto *R = Object->get("result"))
      Result = std::move(*R);
    return Handler.onReply(std::move(*ID), std::move(Result));
  }
  // Params should be given, use null if not.
  json::Value Params = nullptr;
  if (auto *P = Object->get("params"))
    Params = std::move(*P);

  if (ID)
    return Handler.onCall(*Method, std::move(Params), std::move(*ID));
  else
    return Handler.onNotify(*Method, std::move(Params));
}

namespace xpcClosure {
// "owner" of this "closure object" - necessary for propagating connection to
// XPCTransport so it can send messages to the client.
XPCTransport *TransportObject = nullptr;
Transport::MessageHandler *HandlerPtr = nullptr;

void connection_handler(xpc_connection_t clientConnection) {
  xpc_connection_set_target_queue(clientConnection, dispatch_get_main_queue());

  xpc_transaction_begin();

  TransportObject->resetClientConnection(clientConnection);

  xpc_connection_set_event_handler(clientConnection, ^(xpc_object_t message) {
    if (message == XPC_ERROR_CONNECTION_INVALID) {
      // connection is being terminated
      log("Received XPC_ERROR_CONNECTION_INVALID message - returning from the "
          "event_handler.");
      return;
    }

    if (xpc_get_type(message) != XPC_TYPE_DICTIONARY) {
      log("Received XPC message of unknown type - returning from the "
          "event_handler.");
      return;
    }

    const json::Value Doc = xpcToJson(message);
    if (Doc == json::Value(nullptr)) {
      log("XPC message was converted to Null JSON message - returning from the "
          "event_handler.");
      return;
    }

    vlog("<<< {0}\n", Doc);

    if (!TransportObject->handleMessage(std::move(Doc), *HandlerPtr)) {
      log("Received exit notification - cancelling connection.");
      xpc_connection_cancel(xpc_dictionary_get_remote_connection(message));
      xpc_transaction_end();
    }
  });

  xpc_connection_resume(clientConnection);
}
} // namespace xpcClosure

Error XPCTransport::loop(MessageHandler &Handler) {
  assert(xpcClosure::TransportObject == nullptr &&
         "TransportObject has already been set.");
  // This looks scary since lifetime of this (or any) XPCTransport object has
  // to fully contain lifetime of any XPC connection. In practise any Transport
  // object is destroyed only at the end of main() which is always after
  // exit of xpc_main().
  xpcClosure::TransportObject = this;

  assert(xpcClosure::HandlerPtr == nullptr &&
         "HandlerPtr has already been set.");
  xpcClosure::HandlerPtr = &Handler;

  xpc_main(xpcClosure::connection_handler);
  // xpc_main doesn't ever return
  return errorCodeToError(std::make_error_code(std::errc::io_error));
}

} // namespace

namespace clang {
namespace clangd {

std::unique_ptr<Transport> newXPCTransport() {
  return std::make_unique<XPCTransport>();
}

} // namespace clangd
} // namespace clang
