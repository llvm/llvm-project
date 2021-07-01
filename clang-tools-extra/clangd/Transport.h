//===--- Transport.h - sending and receiving LSP messages -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The language server protocol is usually implemented by writing messages as
// JSON-RPC over the stdin/stdout of a subprocess. However other communications
// mechanisms are possible, such as XPC on mac.
//
// The Transport interface allows the mechanism to be replaced, and the JSONRPC
// Transport is the standard implementation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_TRANSPORT_H_
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_TRANSPORT_H_

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace clangd {

// A transport is responsible for maintaining the connection to a client
// application, and reading/writing structured messages to it.
//
// Transports have limited thread safety requirements:
//  - messages will not be sent concurrently
//  - messages MAY be sent while loop() is reading, or its callback is active
class Transport {
public:
  virtual ~Transport() = default;

  // Called by Clangd to send messages to the client.
  virtual void notify(llvm::StringRef Method, llvm::json::Value Params) = 0;
  virtual void call(llvm::StringRef Method, llvm::json::Value Params,
                    llvm::json::Value ID) = 0;
  virtual void reply(llvm::json::Value ID,
                     llvm::Expected<llvm::json::Value> Result) = 0;

  // Implemented by Clangd to handle incoming messages. (See loop() below).
  class MessageHandler {
  public:
    virtual ~MessageHandler() = default;
    // Handler returns true to keep processing messages, or false to shut down.
    virtual bool onNotify(llvm::StringRef Method, llvm::json::Value) = 0;
    virtual bool onCall(llvm::StringRef Method, llvm::json::Value Params,
                        llvm::json::Value ID) = 0;
    virtual bool onReply(llvm::json::Value ID,
                         llvm::Expected<llvm::json::Value> Result) = 0;
  };
  // Called by Clangd to receive messages from the client.
  // The transport should in turn invoke the handler to process messages.
  // If handler returns false, the transport should immediately exit the loop.
  // (This is used to implement the `exit` notification).
  // Otherwise, it returns an error when the transport becomes unusable.
  virtual llvm::Error loop(MessageHandler &) = 0;
};

// Controls the way JSON-RPC messages are encoded (both input and output).
enum JSONStreamStyle {
  // Encoding per the LSP specification, with mandatory Content-Length header.
  Standard,
  // Messages are delimited by a '---' line. Comment lines start with #.
  Delimited
};

// Returns a Transport that speaks JSON-RPC over a pair of streams.
// The input stream must be opened in binary mode.
// If InMirror is set, data read will be echoed to it.
//
// The use of C-style std::FILE* input deserves some explanation.
// Previously, std::istream was used. When a debugger attached on MacOS, the
// process received EINTR, the stream went bad, and clangd exited.
// A retry-on-EINTR loop around reads solved this problem, but caused clangd to
// sometimes hang rather than exit on other OSes. The interaction between
// istreams and signals isn't well-specified, so it's hard to get this right.
// The C APIs seem to be clearer in this respect.
std::unique_ptr<Transport>
newJSONTransport(std::FILE *In, llvm::raw_ostream &Out,
                 llvm::raw_ostream *InMirror, bool Pretty,
                 JSONStreamStyle = JSONStreamStyle::Standard);

#if CLANGD_BUILD_XPC
// Returns a Transport for macOS based on XPC.
// Clangd with this transport is meant to be run as bundled XPC service.
std::unique_ptr<Transport> newXPCTransport();
#endif

} // namespace clangd
} // namespace clang

#endif
