//===--- IncrementalExecutor.cpp - Incremental Execution --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This has the implementation of the base facilities for incremental execution.
//
//===----------------------------------------------------------------------===//

#include "clang/Interpreter/IncrementalExecutor.h"
#include "OrcIncrementalExecutor.h"
#ifdef __EMSCRIPTEN__
#include "Wasm.h"
#endif // __EMSCRIPTEN__

#include "clang/Basic/TargetInfo.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/ToolChain.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include "llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/Debugging/DebuggerSupport.h"
#include "llvm/ExecutionEngine/Orc/EPCDynamicLibrarySearchGenerator.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/MapperJITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/Shared/OrcRTBridge.h"
#include "llvm/ExecutionEngine/Orc/Shared/SimpleRemoteEPCUtils.h"
#include "llvm/ExecutionEngine/Orc/SimpleRemoteEPC.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/TargetParser/Host.h"

#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#ifdef LLVM_ON_UNIX
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace clang {
IncrementalExecutorBuilder::~IncrementalExecutorBuilder() = default;

static llvm::Expected<llvm::orc::JITTargetMachineBuilder>
createJITTargetMachineBuilder(const llvm::Triple &TT) {
  if (TT.getTriple() == llvm::sys::getProcessTriple())
    // This fails immediately if the target backend is not registered
    return llvm::orc::JITTargetMachineBuilder::detectHost();

  // If the target backend is not registered, LLJITBuilder::create() will fail
  return llvm::orc::JITTargetMachineBuilder(TT);
}

static llvm::Expected<std::unique_ptr<llvm::orc::LLJITBuilder>>
createDefaultJITBuilder(llvm::orc::JITTargetMachineBuilder JTMB) {
  auto JITBuilder = std::make_unique<llvm::orc::LLJITBuilder>();
  JITBuilder->setJITTargetMachineBuilder(std::move(JTMB));
  JITBuilder->setPrePlatformSetup([](llvm::orc::LLJIT &J) {
    // Try to enable debugging of JIT'd code (only works with JITLink for
    // ELF and MachO).
    consumeError(llvm::orc::enableDebuggerSupport(J));
    return llvm::Error::success();
  });
  return std::move(JITBuilder);
}

Expected<std::unique_ptr<llvm::jitlink::JITLinkMemoryManager>>
createSharedMemoryManager(llvm::orc::SimpleRemoteEPC &SREPC,
                          unsigned SlabAllocateSize) {
  llvm::orc::SharedMemoryMapper::SymbolAddrs SAs;
  if (auto Err = SREPC.getBootstrapSymbols(
          {{SAs.Instance,
            llvm::orc::rt::ExecutorSharedMemoryMapperServiceInstanceName},
           {SAs.Reserve,
            llvm::orc::rt::ExecutorSharedMemoryMapperServiceReserveWrapperName},
           {SAs.Initialize,
            llvm::orc::rt::
                ExecutorSharedMemoryMapperServiceInitializeWrapperName},
           {SAs.Deinitialize,
            llvm::orc::rt::
                ExecutorSharedMemoryMapperServiceDeinitializeWrapperName},
           {SAs.Release,
            llvm::orc::rt::
                ExecutorSharedMemoryMapperServiceReleaseWrapperName}}))
    return std::move(Err);

  size_t SlabSize;
  if (llvm::Triple(llvm::sys::getProcessTriple()).isOSWindows())
    SlabSize = 1024 * 1024;
  else
    SlabSize = 1024 * 1024 * 1024;

  if (SlabAllocateSize > 0)
    SlabSize = SlabAllocateSize;

  return llvm::orc::MapperJITLinkMemoryManager::CreateWithMapper<
      llvm::orc::SharedMemoryMapper>(SlabSize, SREPC, SAs);
}

static llvm::Expected<
    std::pair<std::unique_ptr<llvm::orc::SimpleRemoteEPC>, uint32_t>>
launchExecutor(llvm::StringRef ExecutablePath, bool UseSharedMemory,
               unsigned SlabAllocateSize, std::function<void()> CustomizeFork) {
#ifndef LLVM_ON_UNIX
  // FIXME: Add support for Windows.
  return llvm::make_error<llvm::StringError>(
      "-" + ExecutablePath + " not supported on non-unix platforms",
      llvm::inconvertibleErrorCode());
#elif !LLVM_ENABLE_THREADS
  // Out of process mode using SimpleRemoteEPC depends on threads.
  return llvm::make_error<llvm::StringError>(
      "-" + ExecutablePath +
          " requires threads, but LLVM was built with "
          "LLVM_ENABLE_THREADS=Off",
      llvm::inconvertibleErrorCode());
#else

  if (!llvm::sys::fs::can_execute(ExecutablePath))
    return llvm::make_error<llvm::StringError>(
        llvm::formatv("Specified executor invalid: {0}", ExecutablePath),
        llvm::inconvertibleErrorCode());

  constexpr int ReadEnd = 0;
  constexpr int WriteEnd = 1;

  // Pipe FDs.
  int ToExecutor[2];
  int FromExecutor[2];

  uint32_t ChildPID;

  // Create pipes to/from the executor..
  if (pipe(ToExecutor) != 0 || pipe(FromExecutor) != 0)
    return llvm::make_error<llvm::StringError>(
        "Unable to create pipe for executor", llvm::inconvertibleErrorCode());

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
      FDSpecifierStr += llvm::utostr(ToExecutor[ReadEnd]);
      FDSpecifierStr += ',';
      FDSpecifierStr += llvm::utostr(FromExecutor[WriteEnd]);
      FDSpecifier = std::make_unique<char[]>(FDSpecifierStr.size() + 1);
      strcpy(FDSpecifier.get(), FDSpecifierStr.c_str());
    }

    char *const Args[] = {ExecutorPath.get(), FDSpecifier.get(), nullptr};
    int RC = execvp(ExecutorPath.get(), Args);
    if (RC != 0) {
      llvm::errs() << "unable to launch out-of-process executor \""
                   << ExecutorPath.get() << "\"\n";
      exit(1);
    }
  }
  // else we're the parent...

  // Close the child ends of the pipes
  close(ToExecutor[ReadEnd]);
  close(FromExecutor[WriteEnd]);

  llvm::orc::SimpleRemoteEPC::Setup S = llvm::orc::SimpleRemoteEPC::Setup();
  if (UseSharedMemory)
    S.CreateMemoryManager =
        [SlabAllocateSize](llvm::orc::SimpleRemoteEPC &EPC) {
          return createSharedMemoryManager(EPC, SlabAllocateSize);
        };

  auto EPCOrErr =
      llvm::orc::SimpleRemoteEPC::Create<llvm::orc::FDSimpleRemoteEPCTransport>(
          std::make_unique<llvm::orc::DynamicThreadPoolTaskDispatcher>(
              std::nullopt),
          std::move(S), FromExecutor[ReadEnd], ToExecutor[WriteEnd]);
  if (!EPCOrErr)
    return EPCOrErr.takeError();
  return std::make_pair(std::move(*EPCOrErr), ChildPID);
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
    return llvm::make_error<llvm::StringError>(
        llvm::formatv("address resolution failed ({0})", strerror(EC)),
        llvm::inconvertibleErrorCode());
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
    return llvm::make_error<llvm::StringError>("invalid hostname",
                                               llvm::inconvertibleErrorCode());

  return SockFD;
}

static llvm::Expected<std::unique_ptr<llvm::orc::SimpleRemoteEPC>>
connectTCPSocket(llvm::StringRef NetworkAddress, bool UseSharedMemory,
                 unsigned SlabAllocateSize) {
#ifndef LLVM_ON_UNIX
  // FIXME: Add TCP support for Windows.
  return llvm::make_error<llvm::StringError>(
      "-" + NetworkAddress + " not supported on non-unix platforms",
      llvm::inconvertibleErrorCode());
#elif !LLVM_ENABLE_THREADS
  // Out of process mode using SimpleRemoteEPC depends on threads.
  return llvm::make_error<llvm::StringError>(
      "-" + NetworkAddress +
          " requires threads, but LLVM was built with "
          "LLVM_ENABLE_THREADS=Off",
      llvm::inconvertibleErrorCode());
#else

  auto CreateErr = [NetworkAddress](Twine Details) {
    return llvm::make_error<llvm::StringError>(
        formatv("Failed to connect TCP socket '{0}': {1}", NetworkAddress,
                Details),
        llvm::inconvertibleErrorCode());
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

  llvm::orc::SimpleRemoteEPC::Setup S = llvm::orc::SimpleRemoteEPC::Setup();
  if (UseSharedMemory)
    S.CreateMemoryManager =
        [SlabAllocateSize](llvm::orc::SimpleRemoteEPC &EPC) {
          return createSharedMemoryManager(EPC, SlabAllocateSize);
        };

  return llvm::orc::SimpleRemoteEPC::Create<
      llvm::orc::FDSimpleRemoteEPCTransport>(
      std::make_unique<llvm::orc::DynamicThreadPoolTaskDispatcher>(
          std::nullopt),
      std::move(S), *SockFD, *SockFD);
#endif
}
#endif // _WIN32

static llvm::Expected<std::unique_ptr<llvm::orc::LLJITBuilder>>
createLLJITBuilder(std::unique_ptr<llvm::orc::ExecutorProcessControl> EPC,
                   llvm::StringRef OrcRuntimePath) {
  auto JTMB = createJITTargetMachineBuilder(EPC->getTargetTriple());
  if (!JTMB)
    return JTMB.takeError();
  auto JB = createDefaultJITBuilder(std::move(*JTMB));
  if (!JB)
    return JB.takeError();

  (*JB)->setExecutorProcessControl(std::move(EPC));
  (*JB)->setPlatformSetUp(
      llvm::orc::ExecutorNativePlatform(OrcRuntimePath.str()));

  return std::move(*JB);
}

static llvm::Expected<
    std::pair<std::unique_ptr<llvm::orc::LLJITBuilder>, uint32_t>>
outOfProcessJITBuilder(const IncrementalExecutorBuilder &IncrExecutorBuilder) {
  std::unique_ptr<llvm::orc::ExecutorProcessControl> EPC;
  uint32_t childPid = -1;
  if (!IncrExecutorBuilder.OOPExecutor.empty()) {
    // Launch an out-of-process executor locally in a child process.
    auto ResultOrErr = launchExecutor(IncrExecutorBuilder.OOPExecutor,
                                      IncrExecutorBuilder.UseSharedMemory,
                                      IncrExecutorBuilder.SlabAllocateSize,
                                      IncrExecutorBuilder.CustomizeFork);
    if (!ResultOrErr)
      return ResultOrErr.takeError();
    childPid = ResultOrErr->second;
    auto EPCOrErr = std::move(ResultOrErr->first);
    EPC = std::move(EPCOrErr);
  } else if (IncrExecutorBuilder.OOPExecutorConnect != "") {
#if LLVM_ON_UNIX && LLVM_ENABLE_THREADS
    auto EPCOrErr = connectTCPSocket(IncrExecutorBuilder.OOPExecutorConnect,
                                     IncrExecutorBuilder.UseSharedMemory,
                                     IncrExecutorBuilder.SlabAllocateSize);
    if (!EPCOrErr)
      return EPCOrErr.takeError();
    EPC = std::move(*EPCOrErr);
#else
    return llvm::make_error<llvm::StringError>(
        "Out-of-process JIT over TCP is not supported on this platform",
        std::error_code());
#endif
  }

  std::unique_ptr<llvm::orc::LLJITBuilder> JB;
  if (EPC) {
    auto JBOrErr =
        createLLJITBuilder(std::move(EPC), IncrExecutorBuilder.OrcRuntimePath);
    if (!JBOrErr)
      return JBOrErr.takeError();
    JB = std::move(*JBOrErr);
  }

  return std::make_pair(std::move(JB), childPid);
}

llvm::Expected<std::unique_ptr<IncrementalExecutor>>
IncrementalExecutorBuilder::create(llvm::orc::ThreadSafeContext &TSC,
                                   const clang::TargetInfo &TI) {
  if (IE)
    return std::move(IE);
  llvm::Triple TT = TI.getTriple();
  if (!TT.isOSWindows() && IsOutOfProcess) {
    if (!JITBuilder) {
      auto ResOrErr = outOfProcessJITBuilder(*this);
      if (!ResOrErr)
        return ResOrErr.takeError();
      JITBuilder = std::move(ResOrErr->first);
      ExecutorPID = ResOrErr->second;
    }
    if (!JITBuilder)
      return llvm::make_error<llvm::StringError>(
          "Operation failed. No LLJITBuilder for out-of-process JIT",
          std::error_code());
  }

  if (!JITBuilder) {
    auto JTMB = createJITTargetMachineBuilder(TT);
    if (!JTMB)
      return JTMB.takeError();
    if (CM)
      JTMB->setCodeModel(CM);
    auto JB = createDefaultJITBuilder(std::move(*JTMB));
    if (!JB)
      return JB.takeError();
    JITBuilder = std::move(*JB);
  }

  llvm::Error Err = llvm::Error::success();
  std::unique_ptr<IncrementalExecutor> Executor;
#ifdef __EMSCRIPTEN__
  Executor = std::make_unique<WasmIncrementalExecutor>(Err);
#else
  Executor = std::make_unique<OrcIncrementalExecutor>(TSC, *JITBuilder, Err);
#endif

  if (Err)
    return std::move(Err);

  return std::move(Executor);
}

llvm::Error IncrementalExecutorBuilder::UpdateOrcRuntimePath(
    const clang::driver::Compilation &C) {
  if (!IsOutOfProcess)
    return llvm::Error::success();

  const clang::driver::Driver &D = C.getDriver();
  const clang::driver::ToolChain &TC = C.getDefaultToolChain();

  llvm::SmallVector<std::string, 2> OrcRTLibNames;

  // Get canonical compiler-rt path
  std::string CompilerRTPath = TC.getCompilerRT(C.getArgs(), "orc_rt");
  llvm::StringRef CanonicalFilename = llvm::sys::path::filename(CompilerRTPath);

  if (CanonicalFilename.empty()) {
    return llvm::make_error<llvm::StringError>(
        "Could not determine OrcRuntime filename from ToolChain",
        llvm::inconvertibleErrorCode());
  }

  OrcRTLibNames.push_back(CanonicalFilename.str());

  // Derive legacy spelling (libclang_rt.orc_rt -> orc_rt)
  llvm::StringRef LegacySuffix = CanonicalFilename;
  if (LegacySuffix.consume_front("libclang_rt.")) {
    OrcRTLibNames.push_back(("lib" + LegacySuffix).str());
  }

  // Extract directory
  llvm::SmallString<256> OrcRTDir(CompilerRTPath);
  llvm::sys::path::remove_filename(OrcRTDir);

  llvm::SmallVector<std::string, 8> triedPaths;

  auto findInDir = [&](llvm::StringRef Dir) -> std::optional<std::string> {
    for (const auto &LibName : OrcRTLibNames) {
      llvm::SmallString<256> FullPath = Dir;
      llvm::sys::path::append(FullPath, LibName);
      if (llvm::sys::fs::exists(FullPath))
        return std::string(FullPath.str());
      triedPaths.push_back(std::string(FullPath.str()));
    }
    return std::nullopt;
  };

  // Try the primary directory first
  if (auto Found = findInDir(OrcRTDir)) {
    OrcRuntimePath = *Found;
    return llvm::Error::success();
  }

  // We want to find the relative path from the Driver to the OrcRTDir
  // to replicate that structure elsewhere if needed.
  llvm::StringRef Rel = OrcRTDir.str();
  if (!Rel.consume_front(llvm::sys::path::parent_path(D.Dir))) {
    return llvm::make_error<llvm::StringError>(
        llvm::formatv("OrcRuntime library path ({0}) is not located within the "
                      "Clang resource directory ({1}). Check your installation "
                      "or provide an explicit path via -resource-dir.",
                      OrcRTDir, D.Dir)
            .str(),
        llvm::inconvertibleErrorCode());
  }

  // Generic Backward Search (Climbing the tree)
  // This is useful for unit tests or relocated toolchains
  llvm::SmallString<256> Cursor(D.Dir); // Start from the driver directory
  while (llvm::sys::path::has_parent_path(Cursor)) {
    Cursor = llvm::sys::path::parent_path(Cursor).str();
    llvm::SmallString<256> Candidate = Cursor;
    llvm::sys::path::append(Candidate, Rel);

    if (auto Found = findInDir(Candidate)) {
      OrcRuntimePath = *Found;
      return llvm::Error::success();
    }

    // Safety check
    if (triedPaths.size() > 32)
      break;
  }

  // Build a helpful error string
  std::string Joined;
  for (size_t i = 0; i < triedPaths.size(); ++i) {
    if (i > 0)
      Joined += "\n  ";
    Joined += triedPaths[i];
  }

  return llvm::make_error<llvm::StringError>(
      llvm::formatv("OrcRuntime library not found. Checked:  {0}",
                    Joined.empty() ? "<none>" : Joined)
          .str(),
      std::make_error_code(std::errc::no_such_file_or_directory));
}

} // end namespace clang
