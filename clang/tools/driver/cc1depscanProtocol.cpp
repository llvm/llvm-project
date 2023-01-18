//===- cc1depscanProtocol.cpp - Communications for -cc1depscan ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "cc1depscanProtocol.h"
#include "clang/Tooling/DependencyScanning/ScanAndUpdateArgs.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/StringSaver.h"

#if LLVM_ON_UNIX
#include <sys/socket.h> // FIXME: Unix-only. Not portable.
#include <sys/types.h>  // FIXME: Unix-only. Not portable.
#include <sys/un.h>     // FIXME: Unix-only. Not portable.

using namespace clang;
using namespace clang::cc1depscand;
using namespace llvm;

static constexpr const char *SocketExtension = ".sock";

int cc1depscand::createSocket() { return ::socket(AF_UNIX, SOCK_STREAM, 0); }

int cc1depscand::acceptSocket(int Socket) {
  sockaddr_un DataAddress;
  socklen_t DataLength;
  return ::accept(Socket, reinterpret_cast<sockaddr *>(&DataAddress),
                  &DataLength);
}

static sockaddr_un configureAddress(StringRef BasePath) {
  sockaddr_un Address;
  SmallString<128> SocketPath = BasePath;
  SocketPath.append(SocketExtension);

  Address.sun_family = AF_UNIX;
  if (SocketPath.size() >= sizeof(Address.sun_path))
    llvm::cantFail(llvm::errorCodeToError(
        std::error_code(ENAMETOOLONG, std::generic_category())));
  ::strncpy(Address.sun_path, SocketPath.c_str(), sizeof(Address.sun_path));
  return Address;
}

void cc1depscand::unlinkBoundSocket(StringRef BasePath) {
  SmallString<128> SocketPath = BasePath;
  SocketPath.append(SocketExtension);
  ::unlink(SocketPath.c_str());
}

int cc1depscand::connectToSocket(StringRef BasePath, int Socket) {
  sockaddr_un Address = configureAddress(BasePath);
  return ::connect(Socket, reinterpret_cast<sockaddr *>(&Address),
                   sizeof(Address));
}
int cc1depscand::bindToSocket(StringRef BasePath, int Socket) {
  sockaddr_un Address = configureAddress(BasePath);
  if (int Failure = ::bind(Socket, reinterpret_cast<sockaddr *>(&Address),
                           sizeof(Address))) {
    if (errno == EADDRINUSE) {
      unlinkBoundSocket(BasePath);
      Failure = ::bind(Socket, reinterpret_cast<sockaddr *>(&Address),
                       sizeof(Address));
    }
    if (Failure)
      return Failure;
  }

  // FIXME: shouldn't compute socket path twice. Also, not sure this is working
  // on crashes.
  SmallString<128> SocketPath = BasePath;
  SocketPath.append(SocketExtension);
  llvm::sys::RemoveFileOnSignal(SocketPath);
  return 0;
}

std::string cc1depscand::getBasePath(StringRef DaemonKey) {
  assert(!DaemonKey.empty() && "Expected valid daemon key");

  // Construct the path.
  SmallString<128> BasePath;
  llvm::sys::path::system_temp_directory(/*ErasedOnReboot=*/true, BasePath);
  llvm::sys::path::append(BasePath, "llvm.depscan", DaemonKey);

  // Ensure null-termination.
  return BasePath.str().str();
}

Expected<OpenSocket> OpenSocket::create(StringRef BasePath) {
  OpenSocket Socket(::socket(AF_UNIX, SOCK_STREAM, 0));
  if (Socket == -1)
    return llvm::errorCodeToError(
        std::error_code(errno, std::generic_category()));
  return std::move(Socket);
}

Expected<ScanDaemon> ScanDaemon::connectToDaemon(StringRef BasePath,
                                                 bool ShouldWait) {
  auto reportError = [BasePath](Error &&E) -> Error {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        Twine("could not connect to scan daemon on path '") + BasePath +
            "': " + toString(std::move(E)));
  };

  Expected<OpenSocket> Socket = OpenSocket::create(BasePath);
  if (!Socket)
    return reportError(Socket.takeError());

  // Wait up to 30 seconds.
  constexpr int MaxWait = 30 * 1000 * 1000;
  int NextBackoff = 0;
  int TotalBackoff = 0;
  while (TotalBackoff < MaxWait) {
    TotalBackoff += NextBackoff;
    if (NextBackoff > 0) {
      if (!ShouldWait)
        break;
      ::usleep(NextBackoff);
    }
    ++NextBackoff;

    // The daemon owns the .pid. Try to connect to the server with the .socket.
    if (!cc1depscand::connectToSocket(BasePath, *Socket))
      return ScanDaemon(std::move(*Socket));

    if (errno != ENOENT && errno != ECONNREFUSED)
      return reportError(llvm::errorCodeToError(
          std::error_code(errno, std::generic_category())));
  }

  return reportError(
      llvm::errorCodeToError(std::error_code(ENOENT, std::generic_category())));
}

Expected<ScanDaemon> ScanDaemon::launchDaemon(StringRef BasePath,
                                              const char *Arg0,
                                              const DepscanSharing &Sharing) {
  std::string BasePathCStr = BasePath.str();
  const char *Args[] = {
      Arg0,
      "-cc1depscand",
      "-run",
      BasePathCStr.c_str(),
  };

  ArrayRef<const char *> InitialArgs = ArrayRef(Args);
  SmallVector<const char *> LaunchArgs(InitialArgs.begin(), InitialArgs.end());

  llvm::BumpPtrAllocator Alloc;
  llvm::StringSaver Saver(Alloc);

  if (Sharing.ShareViaIdentifier) {
    // Invocations that share state via identifier will be isolated from
    // unrelated daemons, so the daemon they share is safe to stay alive longer.
    LaunchArgs.push_back("-long-running");
  }
  LaunchArgs.push_back("-cas-args");
  LaunchArgs.append(Sharing.CASArgs);
  LaunchArgs.push_back(nullptr);

  // Spawn attributes
  posix_spawnattr_t Attrs;
  if (int EC = posix_spawnattr_init(&Attrs))
    return llvm::errorCodeToError(std::error_code(EC, std::generic_category()));
  auto Attrs_cleanup =
      llvm::make_scope_exit([&] { posix_spawnattr_destroy(&Attrs); });

#ifdef POSIX_SPAWN_CLOEXEC_DEFAULT
  // In the spawned process, close all file descriptors that are not explicitly
  // described by the file actions object. This is particularly relevant for
  // llbuild which waits on a "control file handle" and, if inherited, it would
  // cause llbuild to wait for the spawned process to exit.
  // FIXME: This is Darwin-specific extension, perform the same function on
  // non-darwin platforms.
  if (int EC = posix_spawnattr_setflags(&Attrs, POSIX_SPAWN_CLOEXEC_DEFAULT))
    return llvm::errorCodeToError(std::error_code(EC, std::generic_category()));
#endif

  ::pid_t Pid;
  int EC = ::posix_spawn(&Pid, Args[0], /*file_actions=*/nullptr, &Attrs,
                         const_cast<char **>(LaunchArgs.data()),
                         /*envp=*/nullptr);
  if (EC)
    return llvm::errorCodeToError(std::error_code(EC, std::generic_category()));

  return connectToJustLaunchedDaemon(BasePath);
}

Expected<ScanDaemon> ScanDaemon::create(StringRef BasePath, const char *Arg0,
                                        const DepscanSharing &Sharing) {
  if (Expected<ScanDaemon> Daemon = connectToExistingDaemon(BasePath))
    return Daemon;
  else
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

Error ScanDaemon::shakeHands() const {
  cc1depscand::CC1DepScanDProtocol Comms(*this);
  cc1depscand::CC1DepScanDProtocol::ResultKind Result;
  if (auto E = Comms.getResultKind(Result))
    return E;

  if (Result != cc1depscand::CC1DepScanDProtocol::SuccessResult)
    return llvm::errorCodeToError(
        std::error_code(ENOTCONN, std::generic_category()));

  return llvm::Error::success();
}

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

llvm::Error
CC1DepScanDProtocol::putCommand(StringRef WorkingDirectory,
                                ArrayRef<const char *> Args,
                                const DepscanPrefixMapping &Mapping) {
  if (llvm::Error E = putString(WorkingDirectory))
    return E;
  if (llvm::Error E = putArgs(Args))
    return E;
  return putDepscanPrefixMapping(Mapping);
}

llvm::Error CC1DepScanDProtocol::getCommand(llvm::StringSaver &Saver,
                                            StringRef &WorkingDirectory,
                                            SmallVectorImpl<const char *> &Args,
                                            DepscanPrefixMapping &Mapping) {
  if (llvm::Error E = getString(Saver, WorkingDirectory))
    return E;
  if (llvm::Error E = getArgs(Saver, Args))
    return E;
  return getDepscanPrefixMapping(Saver, Mapping);
}

llvm::Error CC1DepScanDProtocol::putDepscanPrefixMapping(
    const DepscanPrefixMapping &Mapping) {
  // Construct the message.
  SmallString<256> FullMapping;
  if (Mapping.NewSDKPath)
    FullMapping.append(*Mapping.NewSDKPath);
  FullMapping.push_back(0);
  if (Mapping.NewToolchainPath)
    FullMapping.append(*Mapping.NewToolchainPath);
  FullMapping.push_back(0);
  for (StringRef Map : Mapping.PrefixMap) {
    FullMapping.append(Map);
    FullMapping.push_back(0);
  }
  return putString(FullMapping);
}

llvm::Error
CC1DepScanDProtocol::getDepscanPrefixMapping(llvm::StringSaver &Saver,
                                             DepscanPrefixMapping &Mapping) {
  StringRef FullMapping;
  if (llvm::Error E = getString(Saver, FullMapping))
    return E;

  // Parse the mapping.
  size_t Count = 0;
  for (auto I = FullMapping.begin(), B = I, E = FullMapping.end(); I != E;
       ++I) {
    if (I != B && I[-1])
      continue; // Wait for null-terminator.
    StringRef Map = I;
    switch (Count++) {
    case 0:
      if (!Map.empty())
        Mapping.NewSDKPath = Map;
      break;
    case 1:
      if (!Map.empty())
        Mapping.NewToolchainPath = Map;
      break;
    default:
      Mapping.PrefixMap.push_back(std::string(Map));
      break;
    }
  }
  return llvm::Error::success();
}

llvm::Error
CC1DepScanDProtocol::getScanResult(llvm::StringSaver &Saver, ResultKind &Result,
                                   StringRef &FailedReason, StringRef &RootID,
                                   SmallVectorImpl<const char *> &Args,
                                   StringRef &DiagnosticOutput) {
  if (Error E = getResultKind(Result))
    return E;

  if (Result == ErrorResult)
    return getString(Saver, FailedReason);

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

llvm::Error CC1DepScanDProtocol::putScanResultFailed(StringRef Reason) {
  if (Error E = putResultKind(ErrorResult))
    return E;
  return putString(Reason);
}

#endif /* LLVM_ON_UNIX */
