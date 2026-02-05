//===- cc1depscanProtocol.cpp - Communications for -cc1depscan ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "cc1depscanProtocol.h"
#include "clang/DependencyScanning/ScanAndUpdateArgs.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/Program.h"
#include <chrono>
#include <cstdlib>
#include <thread>

using namespace clang;
using namespace clang::cc1depscand;
using namespace llvm;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

std::string cc1depscand::getBasePath(StringRef DaemonKey) {
  assert(!DaemonKey.empty() && "Expected valid daemon key");

  // Construct the path.
  SmallString<128> BasePath;
  llvm::sys::path::system_temp_directory(/*ErasedOnReboot=*/true, BasePath);
  llvm::sys::path::append(BasePath, "llvm.depscan", DaemonKey);

  // Ensure null-termination.
  return BasePath.str().str();
}

static constexpr const char *SocketExtension = ".sock";

//===----------------------------------------------------------------------===//
// Protocol Implementation
//===----------------------------------------------------------------------===//

llvm::Error CC1DepScanDProtocol::putArgs(ArrayRef<const char *> Args) {
  // Construct the args block.
  SmallString<256> ArgsBlock;
  for (const char *Arg : Args) {
    ArgsBlock.append(Arg);
    ArgsBlock.push_back(0);
  }
  return putString(ArgsBlock);
}

llvm::Error CC1DepScanDProtocol::getArgs(llvm::StringSaver &Saver,
                                         SmallVectorImpl<const char *> &Args) {
  StringRef ArgsBlock;
  if (llvm::Error E = getString(Saver, ArgsBlock))
    return E;

  // Parse the args block.
  assert(Args.empty());
  for (auto I = ArgsBlock.begin(), B = I, E = ArgsBlock.end(); I != E; ++I)
    if (I == B || !I[-1])
      Args.push_back(I);

  return Error::success();
}

llvm::Error CC1DepScanDProtocol::putCommand(StringRef WorkingDirectory,
                                            ArrayRef<const char *> Args) {
  if (llvm::Error E = putString(WorkingDirectory))
    return E;
  if (llvm::Error E = putArgs(Args))
    return E;
  return llvm::Error::success();
}

llvm::Error
CC1DepScanDProtocol::getCommand(llvm::StringSaver &Saver,
                                StringRef &WorkingDirectory,
                                SmallVectorImpl<const char *> &Args) {
  if (llvm::Error E = getString(Saver, WorkingDirectory))
    return E;
  if (llvm::Error E = getArgs(Saver, Args))
    return E;
  return llvm::Error::success();
}

llvm::Error CC1DepScanDProtocol::putScanResultSuccess(
    StringRef RootID, ArrayRef<const char *> Args, StringRef DiagnosticOutput) {
  if (Error E = putResultKind(SuccessResult))
    return E;
  if (Error E = putString(RootID))
    return E;
  if (Error E = putArgs(Args))
    return E;
  return putString(DiagnosticOutput);
}

llvm::Error
CC1DepScanDProtocol::putScanResultFailed(StringRef Reason,
                                         StringRef DiagnosticOutput) {
  if (Error E = putResultKind(ErrorResult))
    return E;
  if (Error E = putString(Reason))
    return E;
  return putString(DiagnosticOutput);
}

llvm::Error
CC1DepScanDProtocol::getScanResult(llvm::StringSaver &Saver, ResultKind &Result,
                                   StringRef &FailedReason, StringRef &RootID,
                                   SmallVectorImpl<const char *> &Args,
                                   StringRef &DiagnosticOutput) {
  if (Error E = getResultKind(Result))
    return E;

  if (Result == ErrorResult) {
    if (Error E = getString(Saver, FailedReason))
      return E;
    return getString(Saver, DiagnosticOutput);
  }

  if (Result == InvalidResult) {
    FailedReason = "invalid scan result";
    return Error::success();
  }

  if (Error E = getString(Saver, RootID))
    return E;
  if (Error E = getArgs(Saver, Args))
    return E;
  return getString(Saver, DiagnosticOutput);
}

//===----------------------------------------------------------------------===//
// Daemon Connection & Lifecycle
//===----------------------------------------------------------------------===//

Expected<ScanDaemon> ScanDaemon::connectToDaemon(StringRef BasePath,
                                                 bool ShouldWait) {
  // Construct the socket path.
  SmallString<128> SocketPath = BasePath;
  SocketPath.append(SocketExtension);

  auto reportError = [SocketPath](Error &&E) -> Error {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        Twine("could not connect to scan daemon on path '") + SocketPath +
            "': " + toString(std::move(E)));
  };

  // Wait up to 60 seconds.
  constexpr int MaxWait = 60 * 1000 * 1000;
  int NextBackoff = 0;
  int TotalBackoff = 0;
  while (TotalBackoff < MaxWait) {
    TotalBackoff += NextBackoff;
    if (NextBackoff > 0) {
      if (!ShouldWait)
        break;
      std::this_thread::sleep_for(std::chrono::microseconds(NextBackoff));
    }
    ++NextBackoff;

    // Try to connect to the daemon socket.
    Expected<std::unique_ptr<llvm::raw_socket_stream>> Stream =
        llvm::raw_socket_stream::createConnectedUnix(SocketPath);

    if (Stream)
      return ScanDaemon(std::move(*Stream));

    // Check if we should retry based on the error.
    Error E = Stream.takeError();
    std::error_code EC = errorToErrorCode(std::move(E));

    bool ShouldRetry = (EC == std::errc::no_such_file_or_directory ||
                        EC == std::errc::connection_refused);
#ifdef _WIN32
    // On Windows, AF_UNIX may return WSAENETDOWN (10050) or WSAECONNREFUSED
    // (10061) when the socket file exists but the daemon isn't ready yet.
    if (EC.value() == 10050 || EC.value() == 10061)
      ShouldRetry = true;
#endif

    if (!ShouldRetry)
      return reportError(llvm::errorCodeToError(EC));

    // Consume the error and retry.
    llvm::consumeError(llvm::errorCodeToError(EC));
  }

  return llvm::createStringError(
      std::make_error_code(std::errc::timed_out),
      "timeout when connecting to scan daemon on path: '" + BasePath + "'");
}

Expected<ScanDaemon> ScanDaemon::launchDaemon(StringRef BasePath,
                                              const char *Arg0,
                                              const DepscanSharing &Sharing) {
  SmallVector<StringRef> LaunchArgs;
  // First argument must be the program name (argv[0])
  LaunchArgs.push_back(Arg0);
  LaunchArgs.push_back("-cc1depscand");
  LaunchArgs.push_back("-run");
  LaunchArgs.push_back(BasePath);

  if (Sharing.ShareViaIdentifier) {
    // Invocations that share state via identifier will be isolated from
    // unrelated daemons, so the daemon they share is safe to stay alive longer.
    LaunchArgs.push_back("-long-running");
  }
  LaunchArgs.push_back("-cas-args");
  for (const char *Arg : Sharing.CASArgs)
    LaunchArgs.push_back(Arg);

  // Set up environment variables to pass through.
  static constexpr const char *PassThroughEnv[] = {
      "LLVM_CAS_LOG",
      "LLVM_CAS_DISABLE_VALIDATION",
#if defined(_WIN32)
      "SYSTEMROOT",
      "Path",
#endif
  };
  llvm::BumpPtrAllocator Alloc;
  llvm::StringSaver Saver(Alloc);
  SmallVector<StringRef> Env;
  for (const char *Name : PassThroughEnv)
    if (const char *Value = getenv(Name))
      Env.push_back(Saver.save(llvm::Twine(Name) + "=" + Value));

  // Spawn the daemon process without waiting for it.
  std::string ErrMsg;
  bool ExecutionFailed = false;
  std::optional<ArrayRef<StringRef>> EnvOpt =
      Env.empty() ? std::nullopt : std::optional<ArrayRef<StringRef>>(Env);

  llvm::sys::ProcessInfo PI = llvm::sys::ExecuteNoWait(
      Arg0, LaunchArgs, EnvOpt, {}, 0, &ErrMsg, &ExecutionFailed);

  if (ExecutionFailed || PI.Pid == llvm::sys::ProcessInfo::InvalidPid)
    return llvm::createStringError(std::make_error_code(std::errc::invalid_argument),
                                   Twine("failed to launch daemon: ") + ErrMsg);

  // Give the daemon a moment to start up and create the socket before attempting
  // to connect. This reduces spurious connection failures and retries.
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  return connectToJustLaunchedDaemon(BasePath);
}

Error ScanDaemon::shakeHands() {
  cc1depscand::CC1DepScanDProtocol Comms(getStream());
  cc1depscand::CC1DepScanDProtocol::ResultKind Result;
  if (auto E = Comms.getResultKind(Result))
    return E;

  if (Result != cc1depscand::CC1DepScanDProtocol::SuccessResult)
    return llvm::createStringError(std::errc::not_connected,
                                   "handshake failed");

  return llvm::Error::success();
}

Expected<ScanDaemon> ScanDaemon::create(StringRef BasePath, const char *Arg0,
                                        const DepscanSharing &Sharing) {
  Expected<ScanDaemon> Daemon = connectToExistingDaemon(BasePath);
  if (Daemon)
    return Daemon;

  llvm::consumeError(Daemon.takeError()); // FIXME: Sometimes return.
  return launchDaemon(BasePath, Arg0, Sharing);
}

Expected<ScanDaemon>
ScanDaemon::constructAndShakeHands(StringRef BasePath, const char *Arg0,
                                   const DepscanSharing &Sharing) {
  auto Daemon = ScanDaemon::create(BasePath, Arg0, Sharing);
  if (!Daemon)
    return Daemon.takeError();

  // If handshake failed, try relaunch the daemon.
  if (auto E = Daemon->shakeHands()) {
    logAllUnhandledErrors(std::move(E), llvm::errs(),
                          "Restarting daemon due to error: ");

    auto NewDaemon = launchDaemon(BasePath, Arg0, Sharing);
    // If recover failed, return Error.
    if (!NewDaemon)
      return NewDaemon.takeError();
    if (auto NE = NewDaemon->shakeHands())
      return std::move(NE);

    return NewDaemon;
  }

  return Daemon;
}

Expected<ScanDaemon> ScanDaemon::connectToDaemonAndShakeHands(StringRef Path) {
  auto Daemon = ScanDaemon::connectToExistingDaemon(Path);
  if (!Daemon)
    return Daemon.takeError();

  if (auto E = Daemon->shakeHands())
    return std::move(E);

  return Daemon;
}
