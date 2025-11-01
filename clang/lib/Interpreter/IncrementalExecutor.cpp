//===--- IncrementalExecutor.cpp - Incremental Execution --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the class which performs incremental code execution.
//
//===----------------------------------------------------------------------===//

#include "IncrementalExecutor.h"

#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Interpreter/PartialTranslationUnit.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/DebugObjectManagerPlugin.h"
#include "llvm/ExecutionEngine/Orc/Debugging/DebuggerSupport.h"
#include "llvm/ExecutionEngine/Orc/EPCDebugObjectRegistrar.h"
#include "llvm/ExecutionEngine/Orc/EPCDynamicLibrarySearchGenerator.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/MapperJITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/Shared/OrcRTBridge.h"
#include "llvm/ExecutionEngine/Orc/Shared/SimpleRemoteEPCUtils.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/JITLoaderGDB.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/Host.h"

#ifdef LLVM_ON_UNIX
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif // LLVM_ON_UNIX

// Force linking some of the runtimes that helps attaching to a debugger.
LLVM_ATTRIBUTE_USED void linkComponents() {
  llvm::errs() << (void *)&llvm_orc_registerJITLoaderGDBWrapper
               << (void *)&llvm_orc_registerJITLoaderGDBAllocAction;
}

namespace clang {
IncrementalExecutor::IncrementalExecutor(llvm::orc::ThreadSafeContext &TSC)
    : TSCtx(TSC) {}

llvm::Expected<std::unique_ptr<llvm::orc::LLJITBuilder>>
IncrementalExecutor::createDefaultJITBuilder(
    llvm::orc::JITTargetMachineBuilder JTMB) {
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

IncrementalExecutor::IncrementalExecutor(llvm::orc::ThreadSafeContext &TSC,
                                         llvm::orc::LLJITBuilder &JITBuilder,
                                         Interpreter::JITConfig Config,
                                         llvm::Error &Err)
    : TSCtx(TSC), OutOfProcessChildPid(Config.ExecutorPID) {
  using namespace llvm::orc;
  llvm::ErrorAsOutParameter EAO(&Err);

  if (auto JitOrErr = JITBuilder.create())
    Jit = std::move(*JitOrErr);
  else {
    Err = JitOrErr.takeError();
    return;
  }
}

IncrementalExecutor::~IncrementalExecutor() {}

llvm::Error IncrementalExecutor::addModule(PartialTranslationUnit &PTU) {
  llvm::orc::ResourceTrackerSP RT =
      Jit->getMainJITDylib().createResourceTracker();
  ResourceTrackers[&PTU] = RT;

  return Jit->addIRModule(RT, {std::move(PTU.TheModule), TSCtx});
}

llvm::Error IncrementalExecutor::removeModule(PartialTranslationUnit &PTU) {

  llvm::orc::ResourceTrackerSP RT = std::move(ResourceTrackers[&PTU]);
  if (!RT)
    return llvm::Error::success();

  ResourceTrackers.erase(&PTU);
  if (llvm::Error Err = RT->remove())
    return Err;
  return llvm::Error::success();
}

// Clean up the JIT instance.
llvm::Error IncrementalExecutor::cleanUp() {
  // This calls the global dtors of registered modules.
  return Jit->deinitialize(Jit->getMainJITDylib());
}

llvm::Error IncrementalExecutor::runCtors() const {
  return Jit->initialize(Jit->getMainJITDylib());
}

llvm::Expected<llvm::orc::ExecutorAddr>
IncrementalExecutor::getSymbolAddress(llvm::StringRef Name,
                                      SymbolNameKind NameKind) const {
  using namespace llvm::orc;
  auto SO = makeJITDylibSearchOrder({&Jit->getMainJITDylib(),
                                     Jit->getPlatformJITDylib().get(),
                                     Jit->getProcessSymbolsJITDylib().get()});

  ExecutionSession &ES = Jit->getExecutionSession();

  auto SymOrErr =
      ES.lookup(SO, (NameKind == LinkerName) ? ES.intern(Name)
                                             : Jit->mangleAndIntern(Name));
  if (auto Err = SymOrErr.takeError())
    return std::move(Err);
  return SymOrErr->getAddress();
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

llvm::Expected<std::pair<std::unique_ptr<llvm::orc::SimpleRemoteEPC>, uint32_t>>
IncrementalExecutor::launchExecutor(llvm::StringRef ExecutablePath,
                                    bool UseSharedMemory,
                                    unsigned SlabAllocateSize,
                                    std::function<void()> CustomizeFork) {
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

llvm::Expected<std::unique_ptr<llvm::orc::SimpleRemoteEPC>>
IncrementalExecutor::connectTCPSocket(llvm::StringRef NetworkAddress,
                                      bool UseSharedMemory,
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

} // namespace clang
