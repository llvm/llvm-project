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

namespace clang::tooling::cc1modbuildd {

llvm::Expected<std::string>
readBufferFromSocket(llvm::raw_socket_stream &Socket) {
  constexpr unsigned short MAX_BUFFER = 4096;
  constexpr std::chrono::milliseconds TIMEOUT(5000);
  char Buffer[MAX_BUFFER];
  std::string ReturnBuffer;

  ssize_t n = 0;
  while ((n = Socket.read(Buffer, MAX_BUFFER, TIMEOUT)) > 0) {
    ReturnBuffer.append(Buffer, n);
    // Read until \n... encountered which is the last line of a YAML document
    if (ReturnBuffer.find("\n...") != std::string::npos)
      break;
  }

  if (Socket.has_error()) {
    std::error_code EC = Socket.error();
    Socket.clear_error();
    return llvm::make_error<llvm::StringError>("Failed socket read", EC);
  }
  return ReturnBuffer;
}

llvm::Error writeBufferToSocket(llvm::raw_socket_stream &Socket,
                                llvm::StringRef Buffer) {
  Socket << Buffer;
  if (Socket.has_error()) {
    std::error_code EC = Socket.error();
    Socket.clear_error();
    return llvm::make_error<llvm::StringError>("Failed socket write", EC);
  }

  Socket.flush();
  return llvm::Error::success();
}

} // namespace  clang::tooling::cc1modbuildd
