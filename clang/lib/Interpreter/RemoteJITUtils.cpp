//===-- RemoteJITUtils.cpp - Utilities for remote-JITing --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// FIXME: Unify this code with similar functionality in llvm-jitlink.
//
//===----------------------------------------------------------------------===//

#include "clang/Interpreter/RemoteJITUtils.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ExecutionEngine/Orc/DebugObjectManagerPlugin.h"
#include "llvm/ExecutionEngine/Orc/EPCDebugObjectRegistrar.h"
#include "llvm/ExecutionEngine/Orc/EPCDynamicLibrarySearchGenerator.h"
#include "llvm/ExecutionEngine/Orc/MapperJITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/Shared/OrcRTBridge.h"
#include "llvm/ExecutionEngine/Orc/Shared/SimpleRemoteEPCUtils.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/JITLoaderGDB.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#ifdef LLVM_ON_UNIX
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif // LLVM_ON_UNIX

using namespace llvm;
using namespace llvm::orc;

#if LLVM_ON_UNIX
static std::vector<pid_t> LaunchedExecutorPID;
#endif

Expected<uint64_t> getSlabAllocSize(StringRef SizeString) {
  SizeString = SizeString.trim();

  uint64_t Units = 1024;

  if (SizeString.ends_with_insensitive("kb"))
    SizeString = SizeString.drop_back(2).rtrim();
  else if (SizeString.ends_with_insensitive("mb")) {
    Units = 1024 * 1024;
    SizeString = SizeString.drop_back(2).rtrim();
  } else if (SizeString.ends_with_insensitive("gb")) {
    Units = 1024 * 1024 * 1024;
    SizeString = SizeString.drop_back(2).rtrim();
  }

  uint64_t SlabSize = 0;
  if (SizeString.getAsInteger(10, SlabSize))
    return make_error<StringError>("Invalid numeric format for slab size",
                                   inconvertibleErrorCode());

  return SlabSize * Units;
}

Expected<std::unique_ptr<jitlink::JITLinkMemoryManager>>
createSharedMemoryManager(SimpleRemoteEPC &SREPC,
                          StringRef SlabAllocateSizeString) {
  SharedMemoryMapper::SymbolAddrs SAs;
  if (auto Err = SREPC.getBootstrapSymbols(
          {{SAs.Instance, rt::ExecutorSharedMemoryMapperServiceInstanceName},
           {SAs.Reserve,
            rt::ExecutorSharedMemoryMapperServiceReserveWrapperName},
           {SAs.Initialize,
            rt::ExecutorSharedMemoryMapperServiceInitializeWrapperName},
           {SAs.Deinitialize,
            rt::ExecutorSharedMemoryMapperServiceDeinitializeWrapperName},
           {SAs.Release,
            rt::ExecutorSharedMemoryMapperServiceReleaseWrapperName}}))
    return std::move(Err);

#ifdef _WIN32
  size_t SlabSize = 1024 * 1024;
#else
  size_t SlabSize = 1024 * 1024 * 1024;
#endif

  if (!SlabAllocateSizeString.empty()) {
    if (Expected<uint64_t> S = getSlabAllocSize(SlabAllocateSizeString))
      SlabSize = *S;
    else
      return S.takeError();
  }

  return MapperJITLinkMemoryManager::CreateWithMapper<SharedMemoryMapper>(
      SlabSize, SREPC, SAs);
}

// Launches an out-of-process executor for remote JIT. The calling program can
// provide a CustomizeFork callback, which allows it to run custom code in the
// child process before exec. This enables sending custom setup or code to be
// executed in the child (out-of-process) executor.
Expected<std::unique_ptr<SimpleRemoteEPC>>
launchExecutor(StringRef ExecutablePath, bool UseSharedMemory,
               llvm::StringRef SlabAllocateSizeString,
               std::function<void()> CustomizeFork) {
#ifndef LLVM_ON_UNIX
  // FIXME: Add support for Windows.
  return make_error<StringError>("-" + ExecutablePath +
                                     " not supported on non-unix platforms",
                                 inconvertibleErrorCode());
#elif !LLVM_ENABLE_THREADS
  // Out of process mode using SimpleRemoteEPC depends on threads.
  return make_error<StringError>(
      "-" + ExecutablePath +
          " requires threads, but LLVM was built with "
          "LLVM_ENABLE_THREADS=Off",
      inconvertibleErrorCode());
#else

  if (!sys::fs::can_execute(ExecutablePath))
    return make_error<StringError>(
        formatv("Specified executor invalid: {0}", ExecutablePath),
        inconvertibleErrorCode());

  constexpr int ReadEnd = 0;
  constexpr int WriteEnd = 1;

  // Pipe FDs.
  int ToExecutor[2];
  int FromExecutor[2];

  pid_t ChildPID;

  // Create pipes to/from the executor..
  if (pipe(ToExecutor) != 0 || pipe(FromExecutor) != 0)
    return make_error<StringError>("Unable to create pipe for executor",
                                   inconvertibleErrorCode());

  ChildPID = fork();

  if (ChildPID == 0) {
    // In the child...

    // Close the parent ends of the pipes
    close(ToExecutor[WriteEnd]);
    close(FromExecutor[ReadEnd]);

    if (CustomizeFork)
      CustomizeFork();

    // Execute the child process.
    std::unique_ptr<char[]> ExecutorPath, FDSpecifier;
    {
      ExecutorPath = std::make_unique<char[]>(ExecutablePath.size() + 1);
      strcpy(ExecutorPath.get(), ExecutablePath.data());

      std::string FDSpecifierStr("filedescs=");
      FDSpecifierStr += utostr(ToExecutor[ReadEnd]);
      FDSpecifierStr += ',';
      FDSpecifierStr += utostr(FromExecutor[WriteEnd]);
      FDSpecifier = std::make_unique<char[]>(FDSpecifierStr.size() + 1);
      strcpy(FDSpecifier.get(), FDSpecifierStr.c_str());
    }

    char *const Args[] = {ExecutorPath.get(), FDSpecifier.get(), nullptr};
    int RC = execvp(ExecutorPath.get(), Args);
    if (RC != 0) {
      errs() << "unable to launch out-of-process executor \""
             << ExecutorPath.get() << "\"\n";
      exit(1);
    }
  }
  // else we're the parent...

  LaunchedExecutorPID.push_back(ChildPID);

  // Close the child ends of the pipes
  close(ToExecutor[ReadEnd]);
  close(FromExecutor[WriteEnd]);

  SimpleRemoteEPC::Setup S = SimpleRemoteEPC::Setup();
  if (UseSharedMemory)
    S.CreateMemoryManager = [SlabAllocateSizeString](SimpleRemoteEPC &EPC) {
      return createSharedMemoryManager(EPC, SlabAllocateSizeString);
    };

  return SimpleRemoteEPC::Create<FDSimpleRemoteEPCTransport>(
      std::make_unique<DynamicThreadPoolTaskDispatcher>(std::nullopt),
      std::move(S), FromExecutor[ReadEnd], ToExecutor[WriteEnd]);
#endif
}

#if LLVM_ON_UNIX && LLVM_ENABLE_THREADS

static Expected<int> connectTCPSocketImpl(std::string Host,
                                          std::string PortStr) {
  addrinfo *AI;
  addrinfo Hints{};
  Hints.ai_family = AF_INET;
  Hints.ai_socktype = SOCK_STREAM;
  Hints.ai_flags = AI_NUMERICSERV;

  if (int EC = getaddrinfo(Host.c_str(), PortStr.c_str(), &Hints, &AI))
    return make_error<StringError>(
        formatv("address resolution failed ({0})", gai_strerror(EC)),
        inconvertibleErrorCode());
  // Cycle through the returned addrinfo structures and connect to the first
  // reachable endpoint.
  int SockFD;
  addrinfo *Server;
  for (Server = AI; Server != nullptr; Server = Server->ai_next) {
    // socket might fail, e.g. if the address family is not supported. Skip to
    // the next addrinfo structure in such a case.
    if ((SockFD = socket(AI->ai_family, AI->ai_socktype, AI->ai_protocol)) < 0)
      continue;

    // If connect returns null, we exit the loop with a working socket.
    if (connect(SockFD, Server->ai_addr, Server->ai_addrlen) == 0)
      break;

    close(SockFD);
  }
  freeaddrinfo(AI);

  // If we reached the end of the loop without connecting to a valid endpoint,
  // dump the last error that was logged in socket() or connect().
  if (Server == nullptr)
    return make_error<StringError>("invalid hostname",
                                   inconvertibleErrorCode());

  return SockFD;
}
#endif

Expected<std::unique_ptr<SimpleRemoteEPC>>
connectTCPSocket(StringRef NetworkAddress, bool UseSharedMemory,
                 llvm::StringRef SlabAllocateSizeString) {
#ifndef LLVM_ON_UNIX
  // FIXME: Add TCP support for Windows.
  return make_error<StringError>("-" + NetworkAddress +
                                     " not supported on non-unix platforms",
                                 inconvertibleErrorCode());
#elif !LLVM_ENABLE_THREADS
  // Out of process mode using SimpleRemoteEPC depends on threads.
  return make_error<StringError>(
      "-" + NetworkAddress +
          " requires threads, but LLVM was built with "
          "LLVM_ENABLE_THREADS=Off",
      inconvertibleErrorCode());
#else

  auto CreateErr = [NetworkAddress](Twine Details) {
    return make_error<StringError>(
        formatv("Failed to connect TCP socket '{0}': {1}", NetworkAddress,
                Details),
        inconvertibleErrorCode());
  };

  StringRef Host, PortStr;
  std::tie(Host, PortStr) = NetworkAddress.split(':');
  if (Host.empty())
    return CreateErr("Host name for -" + NetworkAddress + " can not be empty");
  if (PortStr.empty())
    return CreateErr("Port number in -" + NetworkAddress + " can not be empty");
  int Port = 0;
  if (PortStr.getAsInteger(10, Port))
    return CreateErr("Port number '" + PortStr + "' is not a valid integer");

  Expected<int> SockFD = connectTCPSocketImpl(Host.str(), PortStr.str());
  if (!SockFD)
    return SockFD.takeError();

  SimpleRemoteEPC::Setup S = SimpleRemoteEPC::Setup();
  if (UseSharedMemory)
    S.CreateMemoryManager = [SlabAllocateSizeString](SimpleRemoteEPC &EPC) {
      return createSharedMemoryManager(EPC, SlabAllocateSizeString);
    };

  return SimpleRemoteEPC::Create<FDSimpleRemoteEPCTransport>(
      std::make_unique<DynamicThreadPoolTaskDispatcher>(std::nullopt),
      std::move(S), *SockFD, *SockFD);
#endif
}

#if LLVM_ON_UNIX

pid_t getLastLaunchedExecutorPID() {
  if (!LaunchedExecutorPID.size())
    return -1;
  return LaunchedExecutorPID.back();
}

pid_t getNthLaunchedExecutorPID(int n) {
  if (n - 1 < 0 || n - 1 >= static_cast<int>(LaunchedExecutorPID.size()))
    return -1;
  return LaunchedExecutorPID.at(n - 1);
}
#endif