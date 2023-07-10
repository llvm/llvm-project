//===------------------------ SocketMsgSupport.cpp ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_socket_stream.h"

#include <memory>
#include <string>

using namespace llvm;

namespace clang::tooling::cc1modbuildd {

Expected<std::unique_ptr<raw_socket_stream>>
connectToSocket(StringRef SocketPath) {

  Expected<std::unique_ptr<raw_socket_stream>> MaybeClient =
      raw_socket_stream::createConnectedUnix(SocketPath);
  if (!MaybeClient)
    return std::move(MaybeClient.takeError());

  return std::move(*MaybeClient);
}

Expected<std::string> readBufferFromSocket(raw_socket_stream &Socket) {

  constexpr unsigned short MAX_BUFFER = 4096;
  char Buffer[MAX_BUFFER];
  std::string ReturnBuffer;

  ssize_t n = 0;
  while ((n = Socket.read(Buffer, MAX_BUFFER)) > 0) {
    ReturnBuffer.append(Buffer, n);
    // Read until \n... encountered which is the last line of a YAML document
    if (ReturnBuffer.find("\n...") != std::string::npos)
      break;
  }

  if (Socket.has_error()) {
    std::error_code EC = Socket.error();
    Socket.clear_error();
    return make_error<StringError>("Failed socket read: ", EC);
  }
  return ReturnBuffer;
}

Error writeBufferToSocket(raw_socket_stream &Socket, StringRef Buffer) {
  Socket << Buffer;

  if (Socket.has_error()) {
    std::error_code EC = Socket.error();
    Socket.clear_error();
    return make_error<StringError>("Failed socket write: ", EC);
  }

  Socket.flush();
  return Error::success();
}

} // namespace  clang::tooling::cc1modbuildd
