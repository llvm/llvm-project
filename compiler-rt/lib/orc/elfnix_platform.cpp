//===- elfnix_platform.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains code required to load the rest of the ELF-on-*IX runtime.
//
//===----------------------------------------------------------------------===//

#include "elfnix_platform.h"
#include "common.h"
#include "compiler.h"
#include "error.h"
#include "jit_dispatch.h"
#include "sections_tracker.h"
#include "wrapper_function_utils.h"

#include <algorithm>
#include <map>
#include <mutex>
#include <sstream>
#include <string_view>
#include <unordered_map>
#include <vector>

using namespace orc_rt;
using namespace orc_rt::elfnix;

// Declare function tags for functions in the JIT process.
ORC_RT_JIT_DISPATCH_TAG(__orc_rt_elfnix_push_initializers_tag)
ORC_RT_JIT_DISPATCH_TAG(__orc_rt_elfnix_symbol_lookup_tag)

// eh-frame registration functions, made available via aliases
// installed by the Platform
extern "C" void __register_frame(const void *);
extern "C" void __deregister_frame(const void *);

extern "C" void
__unw_add_dynamic_eh_frame_section(const void *) ORC_RT_WEAK_IMPORT;
extern "C" void
__unw_remove_dynamic_eh_frame_section(const void *) ORC_RT_WEAK_IMPORT;

namespace {

struct TLSInfoEntry {
  unsigned long Key = 0;
  unsigned long DataAddress = 0;
};

struct TLSDescriptor {
  void (*Resolver)(void *);
  TLSInfoEntry *InfoEntry;
};

class ELFNixPlatformRuntimeState {
private:
  struct AtExitEntry {
    void (*Func)(void *);
    void *Arg;
  };

  using AtExitsVector = std::vector<AtExitEntry>;

  struct PerJITDylibState {
    std::string Name;
    void *Header = nullptr;
    size_t RefCount = 0;
    size_t LinkedAgainstRefCount = 0;
    bool AllowReinitialization = false;
    AtExitsVector AtExits;
    std::vector<PerJITDylibState *> Deps;
    RecordSectionsTracker<void (*)()> RecordedInits;

    bool referenced() const {
      return LinkedAgainstRefCount != 0 || RefCount != 0;
    }
  };

public:
  static void initialize(void *DSOHandle);
  static ELFNixPlatformRuntimeState &get();
  static void destroy();

  ELFNixPlatformRuntimeState(void *DSOHandle);

  // Delete copy and move constructors.
  ELFNixPlatformRuntimeState(const ELFNixPlatformRuntimeState &) = delete;
  ELFNixPlatformRuntimeState &
  operator=(const ELFNixPlatformRuntimeState &) = delete;
  ELFNixPlatformRuntimeState(ELFNixPlatformRuntimeState &&) = delete;
  ELFNixPlatformRuntimeState &operator=(ELFNixPlatformRuntimeState &&) = delete;

  Error registerObjectSections(ELFNixPerObjectSectionsToRegister POSR);
  Error registerJITDylib(std::string &Name, void *Handle);
  Error deregisterJITDylib(void *Handle);
  Error registerInits(ExecutorAddr HeaderAddr,
                      std::vector<ExecutorAddrRange> Inits);
  Error deregisterInits(ExecutorAddr HeaderAddr,
                        std::vector<ExecutorAddrRange> Inits);
  Error deregisterObjectSections(ELFNixPerObjectSectionsToRegister POSR);

  const char *dlerror();
  void *dlopen(std::string_view Name, int Mode);
  int dlclose(void *DSOHandle);
  void *dlsym(void *DSOHandle, std::string_view Symbol);

  int registerAtExit(void (*F)(void *), void *Arg, void *DSOHandle);
  void runAtExits(void *DSOHandle);
  void runAtExits(std::unique_lock<std::recursive_mutex> &JDStateLock,
                  PerJITDylibState &JDS);

  /// Returns the base address of the section containing ThreadData.
  Expected<std::pair<const char *, size_t>>
  getThreadDataSectionFor(const char *ThreadData);

  void *getPlatformJDDSOHandle() { return PlatformJDDSOHandle; }

private:
  PerJITDylibState *getJITDylibStateByHeaderAddr(void *DSOHandle);
  PerJITDylibState *getJITDylibStateByName(std::string_view Path);

  Error registerThreadDataSection(span<const char> ThreadDataSection);

  Expected<ExecutorAddr> lookupSymbolInJITDylib(void *DSOHandle,
                                                std::string_view Symbol);

  Error runInits(std::unique_lock<std::recursive_mutex> &JDStatesLock,
                 PerJITDylibState &JDS);
  Expected<void *> dlopenImpl(std::string_view Path, int Mode);
  Error dlopenFull(std::unique_lock<std::recursive_mutex> &JDStatesLock,
                   PerJITDylibState &JDS);
  Error dlopenInitialize(std::unique_lock<std::recursive_mutex> &JDStatesLock,
                         PerJITDylibState &JDS,
                         ELFNixJITDylibDepInfoMap &DepInfo);
  Error dlcloseImpl(void *DSOHandle);
  Error dlcloseInitialize(std::unique_lock<std::recursive_mutex> &JDStatesLock,
                          PerJITDylibState &JDS);

  static ELFNixPlatformRuntimeState *MOPS;

  void *PlatformJDDSOHandle;

  // Frame registration functions:
  void (*registerEHFrameSection)(const void *) = nullptr;
  void (*deregisterEHFrameSection)(const void *) = nullptr;

  // FIXME: Move to thread-state.
  std::string DLFcnError;

  std::recursive_mutex JDStatesMutex;
  std::unordered_map<void *, PerJITDylibState> JDStates;
  std::unordered_map<std::string, void *> JDNameToHeader;

  std::mutex ThreadDataSectionsMutex;
  std::map<const char *, size_t> ThreadDataSections;
};

ELFNixPlatformRuntimeState *ELFNixPlatformRuntimeState::MOPS = nullptr;

void ELFNixPlatformRuntimeState::initialize(void *DSOHandle) {
  assert(!MOPS && "ELFNixPlatformRuntimeState should be null");
  MOPS = new ELFNixPlatformRuntimeState(DSOHandle);
}

ELFNixPlatformRuntimeState &ELFNixPlatformRuntimeState::get() {
  assert(MOPS && "ELFNixPlatformRuntimeState not initialized");
  return *MOPS;
}

void ELFNixPlatformRuntimeState::destroy() {
  assert(MOPS && "ELFNixPlatformRuntimeState not initialized");
  delete MOPS;
}

ELFNixPlatformRuntimeState::ELFNixPlatformRuntimeState(void *DSOHandle)
    : PlatformJDDSOHandle(DSOHandle) {
  if (__unw_add_dynamic_eh_frame_section &&
      __unw_remove_dynamic_eh_frame_section) {
    registerEHFrameSection = __unw_add_dynamic_eh_frame_section;
    deregisterEHFrameSection = __unw_remove_dynamic_eh_frame_section;
  } else {
    registerEHFrameSection = __register_frame;
    deregisterEHFrameSection = __deregister_frame;
  }
}

Error ELFNixPlatformRuntimeState::registerObjectSections(
    ELFNixPerObjectSectionsToRegister POSR) {
  if (POSR.EHFrameSection.Start)
    registerEHFrameSection(POSR.EHFrameSection.Start.toPtr<const char *>());

  if (POSR.ThreadDataSection.Start) {
    if (auto Err = registerThreadDataSection(
            POSR.ThreadDataSection.toSpan<const char>()))
      return Err;
  }

  return Error::success();
}

Error ELFNixPlatformRuntimeState::deregisterObjectSections(
    ELFNixPerObjectSectionsToRegister POSR) {
  if (POSR.EHFrameSection.Start)
    deregisterEHFrameSection(POSR.EHFrameSection.Start.toPtr<const char *>());

  return Error::success();
}

Error ELFNixPlatformRuntimeState::registerJITDylib(std::string &Name,
                                                   void *Handle) {
  std::lock_guard<std::recursive_mutex> Lock(JDStatesMutex);

  if (JDStates.count(Handle)) {
    std::ostringstream ErrStream;
    ErrStream << "Duplicate JITDylib registration for header " << Handle
              << " (name = " << Name << ")";
    return make_error<StringError>(ErrStream.str());
  }

  if (JDNameToHeader.count(Name)) {
    std::ostringstream ErrStream;
    ErrStream << "Duplicate JITDylib registration for header " << Handle
              << " (header = " << Handle << ")";
    return make_error<StringError>(ErrStream.str());
  }

  auto &JD = JDStates[Handle];
  JD.Header = Handle;
  JD.Name = std::move(Name);
  JDNameToHeader[JD.Name] = Handle;
  return Error::success();
}

Error ELFNixPlatformRuntimeState::deregisterJITDylib(void *Handle) {
  std::lock_guard<std::recursive_mutex> Lock(JDStatesMutex);

  auto I = JDStates.find(Handle);
  if (I == JDStates.end()) {
    std::ostringstream ErrStream;
    ErrStream << "Attempted to deregister unrecognized header " << Handle;
    return make_error<StringError>(ErrStream.str());
  }

  auto J = JDNameToHeader.find(
      std::string(I->second.Name.data(), I->second.Name.size()));
  assert(J != JDNameToHeader.end() &&
         "Missing JDNameToHeader entry for JITDylib");
  JDNameToHeader.erase(J);
  JDStates.erase(I);
  return Error::success();
}

Error ELFNixPlatformRuntimeState::registerInits(
    ExecutorAddr HeaderAddr, std::vector<ExecutorAddrRange> Inits) {
  std::lock_guard<std::recursive_mutex> Lock(JDStatesMutex);
  PerJITDylibState *JDS =
      getJITDylibStateByHeaderAddr(HeaderAddr.toPtr<void *>());

  if (!JDS) {
    std::ostringstream ErrStream;
    ErrStream << "Could not register object platform sections for "
                 "unrecognized header "
              << HeaderAddr.toPtr<void *>();
    return make_error<StringError>(ErrStream.str());
  }

  for (auto &I : Inits) {
    JDS->RecordedInits.add(I.toSpan<void (*)()>());
  }

  return Error::success();
}

Error ELFNixPlatformRuntimeState::deregisterInits(
    ExecutorAddr HeaderAddr, std::vector<ExecutorAddrRange> Inits) {

  std::lock_guard<std::recursive_mutex> Lock(JDStatesMutex);
  PerJITDylibState *JDS =
      getJITDylibStateByHeaderAddr(HeaderAddr.toPtr<void *>());

  if (!JDS) {
    std::ostringstream ErrStream;
    ErrStream << "Could not register object platform sections for unrecognized "
                 "header "
              << HeaderAddr.toPtr<void *>();
    return make_error<StringError>(ErrStream.str());
  }

  for (auto &I : Inits) {
    JDS->RecordedInits.removeIfPresent(I);
  }

  return Error::success();
}

const char *ELFNixPlatformRuntimeState::dlerror() { return DLFcnError.c_str(); }

void *ELFNixPlatformRuntimeState::dlopen(std::string_view Path, int Mode) {
  if (auto H = dlopenImpl(Path, Mode))
    return *H;
  else {
    // FIXME: Make dlerror thread safe.
    DLFcnError = toString(H.takeError());
    return nullptr;
  }
}

int ELFNixPlatformRuntimeState::dlclose(void *DSOHandle) {
  if (auto Err = dlcloseImpl(DSOHandle)) {
    DLFcnError = toString(std::move(Err));
    return -1;
  }
  return 0;
}

void *ELFNixPlatformRuntimeState::dlsym(void *DSOHandle,
                                        std::string_view Symbol) {
  auto Addr = lookupSymbolInJITDylib(DSOHandle, Symbol);
  if (!Addr) {
    DLFcnError = toString(Addr.takeError());
    return 0;
  }

  return Addr->toPtr<void *>();
}

int ELFNixPlatformRuntimeState::registerAtExit(void (*F)(void *), void *Arg,
                                               void *DSOHandle) {
  // FIXME: Handle out-of-memory errors, returning -1 if OOM.
  std::lock_guard<std::recursive_mutex> Lock(JDStatesMutex);
  auto *JDS = getJITDylibStateByHeaderAddr(DSOHandle);
  assert(JDS && "JITDylib state not initialized");
  JDS->AtExits.push_back({F, Arg});
  return 0;
}

void ELFNixPlatformRuntimeState::runAtExits(void *DSOHandle) {
  std::unique_lock<std::recursive_mutex> Lock(JDStatesMutex);
  PerJITDylibState *JDS = getJITDylibStateByHeaderAddr(DSOHandle);

  if (JDS)
    runAtExits(Lock, *JDS);
}

void ELFNixPlatformRuntimeState::runAtExits(
    std::unique_lock<std::recursive_mutex> &JDStateLock,
    PerJITDylibState &JDS) {
  AtExitsVector V = std::move(JDS.AtExits);

  while (!V.empty()) {
    auto &AE = V.back();
    AE.Func(AE.Arg);
    V.pop_back();
  }
}

Expected<std::pair<const char *, size_t>>
ELFNixPlatformRuntimeState::getThreadDataSectionFor(const char *ThreadData) {
  std::lock_guard<std::mutex> Lock(ThreadDataSectionsMutex);
  auto I = ThreadDataSections.upper_bound(ThreadData);
  // Check that we have a valid entry conovering this address.
  if (I == ThreadDataSections.begin())
    return make_error<StringError>("No thread local data section for key");
  I = std::prev(I);
  if (ThreadData >= I->first + I->second)
    return make_error<StringError>("No thread local data section for key");
  return *I;
}

ELFNixPlatformRuntimeState::PerJITDylibState *
ELFNixPlatformRuntimeState::getJITDylibStateByHeaderAddr(void *DSOHandle) {
  auto I = JDStates.find(DSOHandle);
  if (I == JDStates.end())
    return nullptr;

  return &I->second;
}

ELFNixPlatformRuntimeState::PerJITDylibState *
ELFNixPlatformRuntimeState::getJITDylibStateByName(std::string_view Name) {
  // FIXME: Avoid creating string copy here.
  auto I = JDNameToHeader.find(std::string(Name.data(), Name.size()));
  if (I == JDNameToHeader.end())
    return nullptr;
  void *H = I->second;
  auto J = JDStates.find(H);
  assert(J != JDStates.end() &&
         "JITDylib has name map entry but no header map entry");
  return &J->second;
}

Error ELFNixPlatformRuntimeState::registerThreadDataSection(
    span<const char> ThreadDataSection) {
  std::lock_guard<std::mutex> Lock(ThreadDataSectionsMutex);
  auto I = ThreadDataSections.upper_bound(ThreadDataSection.data());
  if (I != ThreadDataSections.begin()) {
    auto J = std::prev(I);
    if (J->first + J->second > ThreadDataSection.data())
      return make_error<StringError>("Overlapping .tdata sections");
  }
  ThreadDataSections.insert(
      I, std::make_pair(ThreadDataSection.data(), ThreadDataSection.size()));
  return Error::success();
}

Expected<ExecutorAddr>
ELFNixPlatformRuntimeState::lookupSymbolInJITDylib(void *DSOHandle,
                                                   std::string_view Sym) {
  Expected<ExecutorAddr> Result((ExecutorAddr()));
  if (auto Err = WrapperFunction<SPSExpected<SPSExecutorAddr>(
          SPSExecutorAddr,
          SPSString)>::call(JITDispatch(&__orc_rt_elfnix_symbol_lookup_tag),
                            Result, ExecutorAddr::fromPtr(DSOHandle), Sym))
    return std::move(Err);
  return Result;
}

Error ELFNixPlatformRuntimeState::runInits(
    std::unique_lock<std::recursive_mutex> &JDStatesLock,
    PerJITDylibState &JDS) {
  std::vector<span<void (*)()>> InitSections;
  InitSections.reserve(JDS.RecordedInits.numNewSections());

  JDS.RecordedInits.processNewSections(
      [&](span<void (*)()> Inits) { InitSections.push_back(Inits); });

  JDStatesLock.unlock();
  for (auto Sec : InitSections)
    for (auto *Init : Sec)
      Init();

  JDStatesLock.lock();

  return Error::success();
}

Expected<void *> ELFNixPlatformRuntimeState::dlopenImpl(std::string_view Path,
                                                        int Mode) {
  std::unique_lock<std::recursive_mutex> Lock(JDStatesMutex);
  PerJITDylibState *JDS = getJITDylibStateByName(Path);

  if (!JDS)
    return make_error<StringError>("No registered JTIDylib for path " +
                                   std::string(Path.data(), Path.size()));

  if (auto Err = dlopenFull(Lock, *JDS))
    return std::move(Err);

  ++JDS->RefCount;

  return JDS->Header;
}

Error ELFNixPlatformRuntimeState::dlopenFull(
    std::unique_lock<std::recursive_mutex> &JDStateLock,
    PerJITDylibState &JDS) {
  Expected<ELFNixJITDylibDepInfoMap> DepInfo((ELFNixJITDylibDepInfoMap()));
  JDStateLock.unlock();
  if (auto Err = WrapperFunction<SPSExpected<SPSELFNixJITDylibDepInfoMap>(
          SPSExecutorAddr)>::
          call(JITDispatch(&__orc_rt_elfnix_push_initializers_tag), DepInfo,
               ExecutorAddr::fromPtr(JDS.Header)))
    return Err;
  JDStateLock.lock();

  if (!DepInfo)
    return DepInfo.takeError();

  if (auto Err = dlopenInitialize(JDStateLock, JDS, *DepInfo))
    return Err;

  if (!DepInfo->empty()) {
    std::ostringstream ErrStream;
    ErrStream << "Encountered unrecognized dep-info key headers "
                 "while processing dlopen of "
              << JDS.Name;
    return make_error<StringError>(ErrStream.str());
  }

  return Error::success();
}

Error ELFNixPlatformRuntimeState::dlopenInitialize(
    std::unique_lock<std::recursive_mutex> &JDStatesLock, PerJITDylibState &JDS,
    ELFNixJITDylibDepInfoMap &DepInfo) {

  auto I = DepInfo.find(ExecutorAddr::fromPtr(JDS.Header));
  if (I == DepInfo.end())
    return Error::success();

  auto Deps = std::move(I->second);
  DepInfo.erase(I);

  std::vector<PerJITDylibState *> OldDeps;
  std::swap(JDS.Deps, OldDeps);
  JDS.Deps.reserve(Deps.size());
  for (auto H : Deps) {
    PerJITDylibState *DepJDS = getJITDylibStateByHeaderAddr(H.toPtr<void *>());
    if (!DepJDS) {
      std::ostringstream ErrStream;
      ErrStream << "Encountered unrecognized dep header " << H.toPtr<void *>()
                << " while initializing " << JDS.Name;
      return make_error<StringError>(ErrStream.str());
    }
    ++DepJDS->LinkedAgainstRefCount;
    if (auto Err = dlopenInitialize(JDStatesLock, *DepJDS, DepInfo))
      return Err;
  }

  if (auto Err = runInits(JDStatesLock, JDS))
    return Err;

  for (auto *DepJDS : OldDeps) {
    --DepJDS->LinkedAgainstRefCount;
    if (!DepJDS->referenced())
      if (auto Err = dlcloseInitialize(JDStatesLock, *DepJDS))
        return Err;
  }
  return Error::success();
}

Error ELFNixPlatformRuntimeState::dlcloseImpl(void *DSOHandle) {

  std::unique_lock<std::recursive_mutex> Lock(JDStatesMutex);
  PerJITDylibState *JDS = getJITDylibStateByHeaderAddr(DSOHandle);

  if (!JDS) {
    std::ostringstream ErrStream;
    ErrStream << "No registered JITDylib for " << DSOHandle;
    return make_error<StringError>(ErrStream.str());
  }

  --JDS->RefCount;

  if (!JDS->referenced())
    return dlcloseInitialize(Lock, *JDS);

  return Error::success();
}

Error ELFNixPlatformRuntimeState::dlcloseInitialize(
    std::unique_lock<std::recursive_mutex> &JDStatesLock,
    PerJITDylibState &JDS) {
  runAtExits(JDStatesLock, JDS);
  JDS.RecordedInits.reset();
  for (auto *DepJDS : JDS.Deps)
    if (!JDS.referenced())
      if (auto Err = dlcloseInitialize(JDStatesLock, *DepJDS))
        return Err;

  return Error::success();
}

class ELFNixPlatformRuntimeTLVManager {
public:
  void *getInstance(const char *ThreadData);

private:
  std::unordered_map<const char *, char *> Instances;
  std::unordered_map<const char *, std::unique_ptr<char[]>> AllocatedSections;
};

void *ELFNixPlatformRuntimeTLVManager::getInstance(const char *ThreadData) {
  auto I = Instances.find(ThreadData);
  if (I != Instances.end())
    return I->second;
  auto TDS =
      ELFNixPlatformRuntimeState::get().getThreadDataSectionFor(ThreadData);
  if (!TDS) {
    __orc_rt_log_error(toString(TDS.takeError()).c_str());
    return nullptr;
  }

  auto &Allocated = AllocatedSections[TDS->first];
  if (!Allocated) {
    Allocated = std::make_unique<char[]>(TDS->second);
    memcpy(Allocated.get(), TDS->first, TDS->second);
  }
  size_t ThreadDataDelta = ThreadData - TDS->first;
  assert(ThreadDataDelta <= TDS->second && "ThreadData outside section bounds");

  char *Instance = Allocated.get() + ThreadDataDelta;
  Instances[ThreadData] = Instance;
  return Instance;
}

void destroyELFNixTLVMgr(void *ELFNixTLVMgr) {
  delete static_cast<ELFNixPlatformRuntimeTLVManager *>(ELFNixTLVMgr);
}

} // end anonymous namespace

//------------------------------------------------------------------------------
//                             JIT entry points
//------------------------------------------------------------------------------

ORC_RT_INTERFACE orc_rt_CWrapperFunctionResult
__orc_rt_elfnix_platform_bootstrap(char *ArgData, size_t ArgSize) {
  return WrapperFunction<SPSError(SPSExecutorAddr)>::handle(
             ArgData, ArgSize,
             [](ExecutorAddr DSOHandle) {
               ELFNixPlatformRuntimeState::initialize(
                   DSOHandle.toPtr<void *>());
               return Error::success();
             })
      .release();
}

ORC_RT_INTERFACE orc_rt_CWrapperFunctionResult
__orc_rt_elfnix_platform_shutdown(char *ArgData, size_t ArgSize) {
  return WrapperFunction<SPSError()>::handle(
             ArgData, ArgSize,
             []() {
               ELFNixPlatformRuntimeState::destroy();
               return Error::success();
             })
      .release();
}

ORC_RT_INTERFACE orc_rt_CWrapperFunctionResult
__orc_rt_elfnix_register_jitdylib(char *ArgData, size_t ArgSize) {
  return WrapperFunction<SPSError(SPSString, SPSExecutorAddr)>::handle(
             ArgData, ArgSize,
             [](std::string &JDName, ExecutorAddr HeaderAddr) {
               return ELFNixPlatformRuntimeState::get().registerJITDylib(
                   JDName, HeaderAddr.toPtr<void *>());
             })
      .release();
}

ORC_RT_INTERFACE orc_rt_CWrapperFunctionResult
__orc_rt_elfnix_deregister_jitdylib(char *ArgData, size_t ArgSize) {
  return WrapperFunction<SPSError(SPSExecutorAddr)>::handle(
             ArgData, ArgSize,
             [](ExecutorAddr HeaderAddr) {
               return ELFNixPlatformRuntimeState::get().deregisterJITDylib(
                   HeaderAddr.toPtr<void *>());
             })
      .release();
}

ORC_RT_INTERFACE orc_rt_CWrapperFunctionResult
__orc_rt_elfnix_register_init_sections(char *ArgData, size_t ArgSize) {
  return WrapperFunction<SPSError(SPSExecutorAddr,
                                  SPSSequence<SPSExecutorAddrRange>)>::
      handle(ArgData, ArgSize,
             [](ExecutorAddr HeaderAddr,
                std::vector<ExecutorAddrRange> &Inits) {
               return ELFNixPlatformRuntimeState::get().registerInits(
                   HeaderAddr, std::move(Inits));
             })
          .release();
}

ORC_RT_INTERFACE orc_rt_CWrapperFunctionResult
__orc_rt_elfnix_deregister_init_sections(char *ArgData, size_t ArgSize) {
  return WrapperFunction<SPSError(SPSExecutorAddr,
                                  SPSSequence<SPSExecutorAddrRange>)>::
      handle(ArgData, ArgSize,
             [](ExecutorAddr HeaderAddr,
                std::vector<ExecutorAddrRange> &Inits) {
               return ELFNixPlatformRuntimeState::get().deregisterInits(
                   HeaderAddr, std::move(Inits));
             })
          .release();
}

/// Wrapper function for registering metadata on a per-object basis.
ORC_RT_INTERFACE orc_rt_CWrapperFunctionResult
__orc_rt_elfnix_register_object_sections(char *ArgData, size_t ArgSize) {
  return WrapperFunction<SPSError(SPSELFNixPerObjectSectionsToRegister)>::
      handle(ArgData, ArgSize,
             [](ELFNixPerObjectSectionsToRegister &POSR) {
               return ELFNixPlatformRuntimeState::get().registerObjectSections(
                   std::move(POSR));
             })
          .release();
}

/// Wrapper for releasing per-object metadat.
ORC_RT_INTERFACE orc_rt_CWrapperFunctionResult
__orc_rt_elfnix_deregister_object_sections(char *ArgData, size_t ArgSize) {
  return WrapperFunction<SPSError(SPSELFNixPerObjectSectionsToRegister)>::
      handle(ArgData, ArgSize,
             [](ELFNixPerObjectSectionsToRegister &POSR) {
               return ELFNixPlatformRuntimeState::get()
                   .deregisterObjectSections(std::move(POSR));
             })
          .release();
}

//------------------------------------------------------------------------------
//                           TLV support
//------------------------------------------------------------------------------

ORC_RT_INTERFACE void *__orc_rt_elfnix_tls_get_addr_impl(TLSInfoEntry *D) {
  auto *TLVMgr = static_cast<ELFNixPlatformRuntimeTLVManager *>(
      pthread_getspecific(D->Key));
  if (!TLVMgr)
    TLVMgr = new ELFNixPlatformRuntimeTLVManager();
  if (pthread_setspecific(D->Key, TLVMgr)) {
    __orc_rt_log_error("Call to pthread_setspecific failed");
    return nullptr;
  }

  return TLVMgr->getInstance(
      reinterpret_cast<char *>(static_cast<uintptr_t>(D->DataAddress)));
}

ORC_RT_INTERFACE ptrdiff_t ___orc_rt_elfnix_tlsdesc_resolver_impl(
    TLSDescriptor *D, const char *ThreadPointer) {
  const char *TLVPtr = reinterpret_cast<const char *>(
      __orc_rt_elfnix_tls_get_addr_impl(D->InfoEntry));
  return TLVPtr - ThreadPointer;
}

ORC_RT_INTERFACE orc_rt_CWrapperFunctionResult
__orc_rt_elfnix_create_pthread_key(char *ArgData, size_t ArgSize) {
  return WrapperFunction<SPSExpected<uint64_t>(void)>::handle(
             ArgData, ArgSize,
             []() -> Expected<uint64_t> {
               pthread_key_t Key;
               if (int Err = pthread_key_create(&Key, destroyELFNixTLVMgr)) {
                 __orc_rt_log_error("Call to pthread_key_create failed");
                 return make_error<StringError>(strerror(Err));
               }
               return static_cast<uint64_t>(Key);
             })
      .release();
}

//------------------------------------------------------------------------------
//                           cxa_atexit support
//------------------------------------------------------------------------------

int __orc_rt_elfnix_cxa_atexit(void (*func)(void *), void *arg,
                               void *dso_handle) {
  return ELFNixPlatformRuntimeState::get().registerAtExit(func, arg,
                                                          dso_handle);
}

int __orc_rt_elfnix_atexit(void (*func)(void *)) {
  auto &PlatformRTState = ELFNixPlatformRuntimeState::get();
  return ELFNixPlatformRuntimeState::get().registerAtExit(
      func, NULL, PlatformRTState.getPlatformJDDSOHandle());
}

void __orc_rt_elfnix_cxa_finalize(void *dso_handle) {
  ELFNixPlatformRuntimeState::get().runAtExits(dso_handle);
}

//------------------------------------------------------------------------------
//                        JIT'd dlfcn alternatives.
//------------------------------------------------------------------------------

const char *__orc_rt_elfnix_jit_dlerror() {
  return ELFNixPlatformRuntimeState::get().dlerror();
}

void *__orc_rt_elfnix_jit_dlopen(const char *path, int mode) {
  return ELFNixPlatformRuntimeState::get().dlopen(path, mode);
}

int __orc_rt_elfnix_jit_dlclose(void *dso_handle) {
  return ELFNixPlatformRuntimeState::get().dlclose(dso_handle);
}

void *__orc_rt_elfnix_jit_dlsym(void *dso_handle, const char *symbol) {
  return ELFNixPlatformRuntimeState::get().dlsym(dso_handle, symbol);
}

//------------------------------------------------------------------------------
//                             ELFNix Run Program
//------------------------------------------------------------------------------

ORC_RT_INTERFACE int64_t __orc_rt_elfnix_run_program(
    const char *JITDylibName, const char *EntrySymbolName, int argc,
    char *argv[]) {
  using MainTy = int (*)(int, char *[]);

  void *H = __orc_rt_elfnix_jit_dlopen(JITDylibName,
                                       orc_rt::elfnix::ORC_RT_RTLD_LAZY);
  if (!H) {
    __orc_rt_log_error(__orc_rt_elfnix_jit_dlerror());
    return -1;
  }

  auto *Main =
      reinterpret_cast<MainTy>(__orc_rt_elfnix_jit_dlsym(H, EntrySymbolName));

  if (!Main) {
    __orc_rt_log_error(__orc_rt_elfnix_jit_dlerror());
    return -1;
  }

  int Result = Main(argc, argv);

  if (__orc_rt_elfnix_jit_dlclose(H) == -1)
    __orc_rt_log_error(__orc_rt_elfnix_jit_dlerror());

  return Result;
}
