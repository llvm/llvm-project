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
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/raw_socket_stream.h"

namespace clang::cc1depscand {

/// Configuration for sharing dependency scanning daemon instances.
struct DepscanSharing {
  bool OnlyShareParent = false;
  bool ShareViaIdentifier = false;
  std::optional<StringRef> Name;
  std::optional<StringRef> Stop;
  std::optional<StringRef> Path;
  SmallVector<const char *> CASArgs;
};

/// Represents an argument edit for automatic argument updates.
struct AutoArgEdit {
  uint32_t Index = -1u;
  StringRef NewArg;
};

/// Get the base path for daemon socket based on a daemon key.
std::string getBasePath(StringRef DaemonKey);

/// Manages a connection to a dependency scanning daemon process.
/// Handles daemon lifecycle (launching, connecting, handshaking) and provides
/// access to the underlying socket stream for communication.
class ScanDaemon {
public:
  static Expected<ScanDaemon> create(StringRef BasePath, const char *Arg0,
                                     const DepscanSharing &Sharing);

  static Expected<ScanDaemon>
  constructAndShakeHands(StringRef BasePath, const char *Arg0,
                         const DepscanSharing &Sharing);

  static Expected<ScanDaemon> connectToDaemonAndShakeHands(StringRef BasePath);

  llvm::Error shakeHands();

  llvm::raw_socket_stream &getStream() { return *Stream; }
  const llvm::raw_socket_stream &getStream() const { return *Stream; }

  ScanDaemon(ScanDaemon &&) = default;
  ScanDaemon &operator=(ScanDaemon &&) = default;

  ScanDaemon(const ScanDaemon &) = delete;
  ScanDaemon &operator=(const ScanDaemon &) = delete;

  // Destructor explicitly closes the socket and clears any errors to prevent
  // fatal error in raw_ostream destructor. On Windows, socket close can trigger
  // errors when the peer has already closed the connection.
  // By calling close() first, we set FD = -1 and ShouldClose = false, so the
  // base class destructor's close path is skipped entirely.
  ~ScanDaemon() {
    if (Stream) {
      Stream->close();
      Stream->clear_error();
    }
  }

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

  explicit ScanDaemon(std::unique_ptr<llvm::raw_socket_stream> Stream)
      : Stream(std::move(Stream)) {}

  std::unique_ptr<llvm::raw_socket_stream> Stream;
};

class CC1DepScanDProtocol {
public:
  llvm::Error getMessage(size_t BytesToRead, llvm::SmallVectorImpl<char> &Bytes) {
    Bytes.clear();
    while (BytesToRead) {
      constexpr size_t BufferSize = 4096;
      char Buffer[BufferSize];
      ssize_t BytesRead = Stream.read(Buffer, std::min(BytesToRead, BufferSize));

      if (BytesRead < 0) {
        std::error_code EC = Stream.error();
        if (EC)
          return llvm::errorCodeToError(EC);
        return llvm::createStringError(std::errc::io_error,
                                       "socket read error");
      }

      if (BytesRead == 0)
        return llvm::createStringError(std::errc::message_size,
                                       "unexpected end of stream");

      Bytes.append(Buffer, Buffer + BytesRead);
      BytesToRead -= BytesRead;
    }
    return llvm::Error::success();
  }

  llvm::Error putMessage(size_t BytesToWrite, const char *Bytes) {
    Stream.write(Bytes, BytesToWrite);
    Stream.flush();

    if (std::error_code EC = Stream.error())
      return llvm::errorCodeToError(EC);

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

  llvm::Error putScanResultFailed(StringRef Reason, StringRef DiagnosticOutput);
  llvm::Error putScanResultSuccess(StringRef RootID,
                                   ArrayRef<const char *> Args,
                                   StringRef DiagnosticOutput);
  llvm::Error getScanResult(llvm::StringSaver &Saver, ResultKind &Result,
                            StringRef &FailedReason, StringRef &RootID,
                            SmallVectorImpl<const char *> &Args,
                            StringRef &DiagnosticOutput);

  explicit CC1DepScanDProtocol(llvm::raw_socket_stream &Stream)
      : Stream(Stream) {}
  CC1DepScanDProtocol() = delete;

private:
  llvm::raw_socket_stream &Stream;
  SmallString<128> Message;
};
}

#endif // LLVM_CLANG_TOOLS_DRIVER_CC1DEPSCANPROTOCOL_H
