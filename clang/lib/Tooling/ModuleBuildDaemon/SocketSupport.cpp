//===------------------------- SocketSupport.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/ModuleBuildDaemon/SocketSupport.h"
#include "clang/Basic/Version.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Tooling/ModuleBuildDaemon/Client.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/BLAKE3.h"

// TODO: Make portable
#if LLVM_ON_UNIX

#include <cerrno>
#include <filesystem>
#include <fstream>
#include <signal.h>
#include <spawn.h>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

Expected<int> cc1modbuildd::createSocket() {
  int FD;
  if ((FD = socket(AF_UNIX, SOCK_STREAM, 0)) == -1) {
    std::string Msg = "socket create error: " + std::string(strerror(errno));
    return createStringError(inconvertibleErrorCode(), Msg);
  }
  return FD;
}

Expected<int> cc1modbuildd::connectToSocket(StringRef SocketPath) {

  Expected<int> MaybeFD = cc1modbuildd::createSocket();
  if (!MaybeFD)
    return std::move(MaybeFD.takeError());

  int FD = std::move(*MaybeFD);

  struct sockaddr_un Addr;
  memset(&Addr, 0, sizeof(Addr));
  Addr.sun_family = AF_UNIX;
  strncpy(Addr.sun_path, SocketPath.str().c_str(), sizeof(Addr.sun_path) - 1);

  if (connect(FD, (struct sockaddr *)&Addr, sizeof(Addr)) == -1) {
    close(FD);
    std::string msg = "socket connect error: " + std::string(strerror(errno));
    return createStringError(inconvertibleErrorCode(), msg);
  }
  return FD;
}

Expected<int> cc1modbuildd::connectAndWriteToSocket(std::string Buffer,
                                                    StringRef SocketPath) {

  Expected<int> MaybeConnectedFD = connectToSocket(SocketPath);
  if (!MaybeConnectedFD)
    return std::move(MaybeConnectedFD.takeError());

  int ConnectedFD = std::move(*MaybeConnectedFD);
  llvm::Error Err = writeToSocket(Buffer, ConnectedFD);
  if (Err)
    return std::move(Err);

  return ConnectedFD;
}

Expected<std::string> cc1modbuildd::readFromSocket(int FD) {

  const size_t BUFFER_SIZE = 4096;
  std::vector<char> Buffer(BUFFER_SIZE);
  size_t TotalBytesRead = 0;

  ssize_t n;
  while ((n = read(FD, Buffer.data() + TotalBytesRead,
                   Buffer.size() - TotalBytesRead)) > 0) {

    TotalBytesRead += n;
    // Read until ...\n encountered (last line of YAML document)
    if (std::string(&Buffer[TotalBytesRead - 4], 4) == "...\n")
      break;
    if (Buffer.size() - TotalBytesRead < BUFFER_SIZE)
      Buffer.resize(Buffer.size() + BUFFER_SIZE);
  }

  if (n < 0) {
    std::string Msg = "socket read error: " + std::string(strerror(errno));
    return llvm::make_error<StringError>(Msg, inconvertibleErrorCode());
  }
  if (n == 0)
    return llvm::make_error<StringError>("EOF", inconvertibleErrorCode());
  return std::string(Buffer.begin(), Buffer.end());
}

llvm::Error cc1modbuildd::writeToSocket(std::string Buffer, int WriteFD) {

  ssize_t BytesToWrite = static_cast<ssize_t>(Buffer.size());
  const char *Bytes = Buffer.c_str();

  while (BytesToWrite) {
    ssize_t BytesWritten = write(WriteFD, Bytes, BytesToWrite);
    if (BytesWritten == -1) {
      std::string Msg = "socket write error: " + std::string(strerror(errno));
      return llvm::make_error<StringError>(Msg, inconvertibleErrorCode());
    }

    if (!BytesWritten || BytesWritten > BytesToWrite)
      return llvm::errorCodeToError(
          std::error_code(EIO, std::generic_category()));

    BytesToWrite -= BytesWritten;
    Bytes += BytesWritten;
  }
  return llvm::Error::success();
}

#endif // LLVM_ON_UNIX
