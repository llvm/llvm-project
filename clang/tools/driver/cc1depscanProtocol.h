//===- cc1depscanProtocol.h - Communications for -cc1depscan --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_DRIVER_CC1DEPSCANPROTOCOL_H
#define LLVM_CLANG_TOOLS_DRIVER_CC1DEPSCANPROTOCOL_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/StringSaver.h"

#if LLVM_ON_UNIX
#include <spawn.h>      // FIXME: Unix-only. Not portable.
#include <sys/socket.h> // FIXME: Unix-only. Not portable.
#include <sys/types.h>  // FIXME: Unix-only. Not portable.
#include <sys/un.h>     // FIXME: Unix-only. Not portable.
#include <unistd.h>     // FIXME: Unix-only. Not portable.

namespace clang::cc1depscand {
struct DepscanSharing {
  bool OnlyShareParent = false;
  bool ShareViaIdentifier = false;
  std::optional<StringRef> Name;
  std::optional<StringRef> Stop;
  std::optional<StringRef> Path;
  SmallVector<const char *> CASArgs;
};

struct AutoArgEdit {
  uint32_t Index = -1u;
  StringRef NewArg;
};

std::string getBasePath(StringRef DaemonKey);

int createSocket();
int connectToSocket(StringRef BasePath, int Socket);
int bindToSocket(StringRef BasePath, int Socket);
void unlinkBoundSocket(StringRef BasePath);
int acceptSocket(int Socket);

class CC1DepScanDProtocol {
public:
  llvm::Error getMessage(size_t BytesToRead, SmallVectorImpl<char> &Bytes) {
    Bytes.clear();
    while (BytesToRead) {
      constexpr size_t BufferSize = 4096;
      char Buffer[BufferSize];
      size_t BytesRead = ::read(Socket, static_cast<void *>(Buffer),
                                std::min(BytesToRead, BufferSize));
      if (BytesRead == -1ull)
        return llvm::errorCodeToError(
            std::error_code(errno, std::generic_category()));
      Bytes.append(Buffer, Buffer + BytesRead);

      if (!BytesRead || BytesRead > BytesToRead)
        return llvm::errorCodeToError(
            std::error_code(EMSGSIZE, std::generic_category()));
      BytesToRead -= BytesRead;
    }
    return llvm::Error::success();
  }

  llvm::Error putMessage(size_t BytesToWrite, const char *Bytes) {
    while (BytesToWrite) {
      size_t BytesWritten = ::write(Socket, Bytes, BytesToWrite);
      if (BytesWritten == -1ull)
        return llvm::errorCodeToError(
            std::error_code(errno, std::generic_category()));

      if (!BytesWritten || BytesWritten > BytesToWrite)
        return llvm::errorCodeToError(
            std::error_code(EMSGSIZE, std::generic_category()));
      BytesToWrite -= BytesWritten;
      Bytes += BytesWritten;
    }
    return llvm::Error::success();
  }

  template <class NumberT> llvm::Error getNumber(NumberT &Number) {
    // FIXME: Assumes endianness matches.
    if (llvm::Error E = getMessage(sizeof(Number), Message))
      return E;
    ::memcpy(&Number, Message.begin(), sizeof(Number));
    return llvm::Error::success();
  }
  template <class NumberT> llvm::Error putNumber(NumberT Number) {
    // FIXME: Assumes endianness matches.
    return putMessage(sizeof(Number), reinterpret_cast<const char *>(&Number));
  }

  enum ResultKind {
    ErrorResult,
    SuccessResult,
    InvalidResult, // Last...
  };
  llvm::Error putResultKind(ResultKind Result) {
    return putNumber(unsigned(Result));
  }
  llvm::Error getResultKind(ResultKind &Result) {
    unsigned Read;
    if (llvm::Error E = getNumber(Read))
      return E;
    if (Read >= InvalidResult)
      Result = InvalidResult;
    else
      Result = ResultKind(Read);
    return llvm::Error::success();
  }

  /// Read a string. \p String will be a null-terminated string owned by \p
  /// Saver.
  llvm::Error getString(llvm::StringSaver &Saver, StringRef &String) {
    size_t Length;
    if (llvm::Error E = getNumber(Length))
      return E;
    if (llvm::Error E = getMessage(Length, Message))
      return E;
    String = Saver.save(StringRef(Message));
    return llvm::Error::success();
  }
  llvm::Error putString(StringRef String) {
    if (llvm::Error E = putNumber(String.size()))
      return E;
    return putMessage(String.size(), String.begin());
  }

  llvm::Error putArgs(ArrayRef<const char *> Args);
  llvm::Error getArgs(llvm::StringSaver &Saver,
                      SmallVectorImpl<const char *> &Args);

  llvm::Error putCommand(StringRef WorkingDirectory,
                         ArrayRef<const char *> Args);
  llvm::Error getCommand(llvm::StringSaver &Saver, StringRef &WorkingDirectory,
                         SmallVectorImpl<const char *> &Args);

  llvm::Error putScanResultFailed(StringRef Reason);
  llvm::Error putScanResultSuccess(StringRef RootID,
                                   ArrayRef<const char *> Args,
                                   StringRef DiagnosticOutput);
  llvm::Error getScanResult(llvm::StringSaver &Saver, ResultKind &Result,
                            StringRef &FailedReason, StringRef &RootID,
                            SmallVectorImpl<const char *> &Args,
                            StringRef &DiagnosticOutput);

  explicit CC1DepScanDProtocol(int Socket) : Socket(Socket) {}
  CC1DepScanDProtocol() = delete;

private:
  int Socket;
  SmallString<128> Message;
};

class FileDescriptor {
public:
  operator int() const { return FD; }

  FileDescriptor() = default;
  explicit FileDescriptor(int FD) : FD(FD) {}

  FileDescriptor(FileDescriptor &&X) : FD(X) { X.FD = -1; }
  FileDescriptor &operator=(FileDescriptor &&X) {
    close();
    FD = X.FD;
    X.FD = -1;
    return *this;
  }

  FileDescriptor(const FileDescriptor &) = delete;
  FileDescriptor &operator=(const FileDescriptor &) = delete;

  ~FileDescriptor() { close(); }

private:
  void close() {
    if (FD == -1)
      return;
    ::close(FD);
    FD = -1;
  }
  int FD = -1;
};

class OpenSocket : public FileDescriptor {
public:
  static Expected<OpenSocket> create(StringRef BasePath);

  OpenSocket() = delete;

  OpenSocket(OpenSocket &&) = default;
  OpenSocket &operator=(OpenSocket &&Socket) = default;

private:
  explicit OpenSocket(int FD) : FileDescriptor(FD) {}
};

class ScanDaemon : public OpenSocket {
public:
  static Expected<ScanDaemon> create(StringRef BasePath, const char *Arg0,
                                     const DepscanSharing &Sharing);

  static Expected<ScanDaemon>
  constructAndShakeHands(StringRef BasePath, const char *Arg0,
                         const DepscanSharing &Sharing);

  static Expected<ScanDaemon> connectToDaemonAndShakeHands(StringRef BasePath);

  llvm::Error shakeHands() const;

private:
  static Expected<ScanDaemon> launchDaemon(StringRef BasePath, const char *Arg0,
                                           const DepscanSharing &Sharing);
  static Expected<ScanDaemon> connectToDaemon(StringRef BasePath,
                                              bool ShouldWait);
  static Expected<ScanDaemon> connectToExistingDaemon(StringRef BasePath) {
    return connectToDaemon(BasePath, /*ShouldWait=*/false);
  }
  static Expected<ScanDaemon> connectToJustLaunchedDaemon(StringRef BasePath) {
    return connectToDaemon(BasePath, /*ShouldWait=*/true);
  }

  explicit ScanDaemon(OpenSocket S) : OpenSocket(std::move(S)) {}
};

} // namespace clang::cc1depscand

#endif /* LLVM_ON_UNIX */
#endif // LLVM_CLANG_TOOLS_DRIVER_CC1DEPSCANPROTOCOL_H
