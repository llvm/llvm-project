//===- macho_platform.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains code required to load the rest of the MachO runtime.
//
//===----------------------------------------------------------------------===//

#include "macho_platform.h"
#include "common.h"
#include "error.h"
#include "wrapper_function_utils.h"

#include <map>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <vector>

using namespace __orc_rt;
using namespace __orc_rt::macho;

// Declare function tags for functions in the JIT process.
ORC_RT_JIT_DISPATCH_TAG(__orc_rt_macho_get_initializers_tag)
ORC_RT_JIT_DISPATCH_TAG(__orc_rt_macho_get_deinitializers_tag)
ORC_RT_JIT_DISPATCH_TAG(__orc_rt_macho_symbol_lookup_tag)

// Objective-C types.
struct objc_class;
struct objc_image_info;
struct objc_object;
struct objc_selector;

using Class = objc_class *;
using id = objc_object *;
using SEL = objc_selector *;

// Objective-C registration functions.
// These are weakly imported. If the Objective-C runtime has not been loaded
// then code containing Objective-C sections will generate an error.
extern "C" id objc_msgSend(id, SEL, ...) ORC_RT_WEAK_IMPORT;
extern "C" Class objc_readClassPair(Class,
                                    const objc_image_info *) ORC_RT_WEAK_IMPORT;
extern "C" SEL sel_registerName(const char *) ORC_RT_WEAK_IMPORT;

// Swift types.
class ProtocolRecord;
class ProtocolConformanceRecord;
class TypeMetadataRecord;

extern "C" void
swift_registerProtocols(const ProtocolRecord *begin,
                        const ProtocolRecord *end) ORC_RT_WEAK_IMPORT;

extern "C" void swift_registerProtocolConformances(
    const ProtocolConformanceRecord *begin,
    const ProtocolConformanceRecord *end) ORC_RT_WEAK_IMPORT;

extern "C" void swift_registerTypeMetadataRecords(
    const TypeMetadataRecord *begin,
    const TypeMetadataRecord *end) ORC_RT_WEAK_IMPORT;

namespace {

Error validatePointerSectionExtent(const char *SectionName,
                                   const ExecutorAddrRange &SE) {
  if (SE.size().getValue() % sizeof(uintptr_t)) {
    std::ostringstream ErrMsg;
    ErrMsg << std::hex << "Size of " << SectionName << " 0x"
           << SE.Start.getValue() << " -- 0x" << SE.End.getValue()
           << " is not a pointer multiple";
    return make_error<StringError>(ErrMsg.str());
  }
  return Error::success();
}

Error registerObjCSelectors(
    const std::vector<ExecutorAddrRange> &ObjCSelRefsSections,
    const MachOJITDylibInitializers &MOJDIs) {

  if (ORC_RT_UNLIKELY(!sel_registerName))
    return make_error<StringError>("sel_registerName is not available");

  for (const auto &ObjCSelRefs : ObjCSelRefsSections) {

    if (auto Err = validatePointerSectionExtent("__objc_selrefs", ObjCSelRefs))
      return Err;

    for (uintptr_t &SelEntry : ObjCSelRefs.toSpan<uintptr_t>()) {
      const char *SelName = reinterpret_cast<const char *>(SelEntry);
      auto Sel = sel_registerName(SelName);
      *reinterpret_cast<SEL *>(&SelEntry) = Sel;
    }
  }

  return Error::success();
}

Error registerObjCClasses(
    const std::vector<ExecutorAddrRange> &ObjCClassListSections,
    const MachOJITDylibInitializers &MOJDIs) {

  if (ObjCClassListSections.empty())
    return Error::success();

  if (ORC_RT_UNLIKELY(!objc_msgSend))
    return make_error<StringError>("objc_msgSend is not available");
  if (ORC_RT_UNLIKELY(!objc_readClassPair))
    return make_error<StringError>("objc_readClassPair is not available");

  struct ObjCClassCompiled {
    void *Metaclass;
    void *Parent;
    void *Cache1;
    void *Cache2;
    void *Data;
  };

  auto *ImageInfo =
      MOJDIs.ObjCImageInfoAddress.toPtr<const objc_image_info *>();
  auto ClassSelector = sel_registerName("class");

  for (const auto &ObjCClassList : ObjCClassListSections) {

    if (auto Err =
            validatePointerSectionExtent("__objc_classlist", ObjCClassList))
      return Err;

    for (uintptr_t ClassPtr : ObjCClassList.toSpan<uintptr_t>()) {
      auto *Cls = reinterpret_cast<Class>(ClassPtr);
      auto *ClassCompiled = reinterpret_cast<ObjCClassCompiled *>(ClassPtr);
      objc_msgSend(reinterpret_cast<id>(ClassCompiled->Parent), ClassSelector);
      auto Registered = objc_readClassPair(Cls, ImageInfo);

      // FIXME: Improve diagnostic by reporting the failed class's name.
      if (Registered != Cls)
        return make_error<StringError>("Unable to register Objective-C class");
    }
  }
  return Error::success();
}

Error registerSwift5Protocols(
    const std::vector<ExecutorAddrRange> &Swift5ProtocolSections,
    const MachOJITDylibInitializers &MOJDIs) {

  if (ORC_RT_UNLIKELY(!Swift5ProtocolSections.empty() &&
                      !swift_registerProtocols))
    return make_error<StringError>("swift_registerProtocols is not available");

  for (const auto &Swift5Protocols : Swift5ProtocolSections)
    swift_registerProtocols(
        Swift5Protocols.Start.toPtr<const ProtocolRecord *>(),
        Swift5Protocols.End.toPtr<const ProtocolRecord *>());

  return Error::success();
}

Error registerSwift5ProtocolConformances(
    const std::vector<ExecutorAddrRange> &Swift5ProtocolConformanceSections,
    const MachOJITDylibInitializers &MOJDIs) {

  if (ORC_RT_UNLIKELY(!Swift5ProtocolConformanceSections.empty() &&
                      !swift_registerProtocolConformances))
    return make_error<StringError>(
        "swift_registerProtocolConformances is not available");

  for (const auto &ProtoConfSec : Swift5ProtocolConformanceSections)
    swift_registerProtocolConformances(
        ProtoConfSec.Start.toPtr<const ProtocolConformanceRecord *>(),
        ProtoConfSec.End.toPtr<const ProtocolConformanceRecord *>());

  return Error::success();
}

Error registerSwift5Types(const std::vector<ExecutorAddrRange> &Sections,
                          const MachOJITDylibInitializers &MOJDIs) {

  if (ORC_RT_UNLIKELY(!Sections.empty() && !swift_registerTypeMetadataRecords))
    return make_error<StringError>(
        "swift_registerTypeMetadataRecords is not available");

  for (const auto &Section : Sections)
    swift_registerTypeMetadataRecords(
        Section.Start.toPtr<const TypeMetadataRecord *>(),
        Section.End.toPtr<const TypeMetadataRecord *>());

  return Error::success();
}

Error runModInits(const std::vector<ExecutorAddrRange> &ModInitsSections,
                  const MachOJITDylibInitializers &MOJDIs) {

  for (const auto &ModInits : ModInitsSections) {
    if (auto Err = validatePointerSectionExtent("__mod_inits", ModInits))
      return Err;

    using InitFunc = void (*)();
    for (auto *Init : ModInits.toSpan<InitFunc>())
      (*Init)();
  }

  return Error::success();
}

struct TLVDescriptor {
  void *(*Thunk)(TLVDescriptor *) = nullptr;
  unsigned long Key = 0;
  unsigned long DataAddress = 0;
};

class MachOPlatformRuntimeState {
private:
  struct AtExitEntry {
    void (*Func)(void *);
    void *Arg;
  };

  using AtExitsVector = std::vector<AtExitEntry>;

  struct PerJITDylibState {
    void *Header = nullptr;
    size_t RefCount = 0;
    bool AllowReinitialization = false;
    AtExitsVector AtExits;
  };

public:
  static void initialize();
  static MachOPlatformRuntimeState &get();
  static void destroy();

  MachOPlatformRuntimeState() = default;

  // Delete copy and move constructors.
  MachOPlatformRuntimeState(const MachOPlatformRuntimeState &) = delete;
  MachOPlatformRuntimeState &
  operator=(const MachOPlatformRuntimeState &) = delete;
  MachOPlatformRuntimeState(MachOPlatformRuntimeState &&) = delete;
  MachOPlatformRuntimeState &operator=(MachOPlatformRuntimeState &&) = delete;

  Error registerThreadDataSection(span<const char> ThreadDataSec);
  Error deregisterThreadDataSection(span<const char> ThreadDataSec);

  const char *dlerror();
  void *dlopen(string_view Name, int Mode);
  int dlclose(void *DSOHandle);
  void *dlsym(void *DSOHandle, string_view Symbol);

  int registerAtExit(void (*F)(void *), void *Arg, void *DSOHandle);
  void runAtExits(void *DSOHandle);

  /// Returns the base address of the section containing ThreadData.
  Expected<std::pair<const char *, size_t>>
  getThreadDataSectionFor(const char *ThreadData);

private:
  PerJITDylibState *getJITDylibStateByHeaderAddr(void *DSOHandle);
  PerJITDylibState *getJITDylibStateByName(string_view Path);
  PerJITDylibState &getOrCreateJITDylibState(MachOJITDylibInitializers &MOJDIs);

  Expected<ExecutorAddr> lookupSymbolInJITDylib(void *DSOHandle,
                                                string_view Symbol);

  Expected<MachOJITDylibInitializerSequence>
  getJITDylibInitializersByName(string_view Path);
  Expected<void *> dlopenInitialize(string_view Path, int Mode);
  Error initializeJITDylib(MachOJITDylibInitializers &MOJDIs);

  static MachOPlatformRuntimeState *MOPS;

  using InitSectionHandler =
      Error (*)(const std::vector<ExecutorAddrRange> &Sections,
                const MachOJITDylibInitializers &MOJDIs);
  const std::vector<std::pair<const char *, InitSectionHandler>> InitSections =
      {{"__DATA,__objc_selrefs", registerObjCSelectors},
       {"__DATA,__objc_classlist", registerObjCClasses},
       {"__TEXT,__swift5_protos", registerSwift5Protocols},
       {"__TEXT,__swift5_proto", registerSwift5ProtocolConformances},
       {"__TEXT,__swift5_types", registerSwift5Types},
       {"__DATA,__mod_init_func", runModInits}};

  // FIXME: Move to thread-state.
  std::string DLFcnError;

  std::recursive_mutex JDStatesMutex;
  std::unordered_map<void *, PerJITDylibState> JDStates;
  std::unordered_map<std::string, void *> JDNameToHeader;

  std::mutex ThreadDataSectionsMutex;
  std::map<const char *, size_t> ThreadDataSections;
};

MachOPlatformRuntimeState *MachOPlatformRuntimeState::MOPS = nullptr;

void MachOPlatformRuntimeState::initialize() {
  assert(!MOPS && "MachOPlatformRuntimeState should be null");
  MOPS = new MachOPlatformRuntimeState();
}

MachOPlatformRuntimeState &MachOPlatformRuntimeState::get() {
  assert(MOPS && "MachOPlatformRuntimeState not initialized");
  return *MOPS;
}

void MachOPlatformRuntimeState::destroy() {
  assert(MOPS && "MachOPlatformRuntimeState not initialized");
  delete MOPS;
}

Error MachOPlatformRuntimeState::registerThreadDataSection(
    span<const char> ThreadDataSection) {
  std::lock_guard<std::mutex> Lock(ThreadDataSectionsMutex);
  auto I = ThreadDataSections.upper_bound(ThreadDataSection.data());
  if (I != ThreadDataSections.begin()) {
    auto J = std::prev(I);
    if (J->first + J->second > ThreadDataSection.data())
      return make_error<StringError>("Overlapping __thread_data sections");
  }
  ThreadDataSections.insert(
      I, std::make_pair(ThreadDataSection.data(), ThreadDataSection.size()));
  return Error::success();
}

Error MachOPlatformRuntimeState::deregisterThreadDataSection(
    span<const char> ThreadDataSection) {
  std::lock_guard<std::mutex> Lock(ThreadDataSectionsMutex);
  auto I = ThreadDataSections.find(ThreadDataSection.data());
  if (I == ThreadDataSections.end())
    return make_error<StringError>("Attempt to deregister unknown thread data "
                                   "section");
  ThreadDataSections.erase(I);
  return Error::success();
}

const char *MachOPlatformRuntimeState::dlerror() { return DLFcnError.c_str(); }

void *MachOPlatformRuntimeState::dlopen(string_view Path, int Mode) {
  std::lock_guard<std::recursive_mutex> Lock(JDStatesMutex);

  // Use fast path if all JITDylibs are already loaded and don't require
  // re-running initializers.
  if (auto *JDS = getJITDylibStateByName(Path)) {
    if (!JDS->AllowReinitialization) {
      ++JDS->RefCount;
      return JDS->Header;
    }
  }

  auto H = dlopenInitialize(Path, Mode);
  if (!H) {
    DLFcnError = toString(H.takeError());
    return nullptr;
  }

  return *H;
}

int MachOPlatformRuntimeState::dlclose(void *DSOHandle) {
  runAtExits(DSOHandle);
  return 0;
}

void *MachOPlatformRuntimeState::dlsym(void *DSOHandle, string_view Symbol) {
  auto Addr = lookupSymbolInJITDylib(DSOHandle, Symbol);
  if (!Addr) {
    DLFcnError = toString(Addr.takeError());
    return 0;
  }

  return Addr->toPtr<void *>();
}

int MachOPlatformRuntimeState::registerAtExit(void (*F)(void *), void *Arg,
                                              void *DSOHandle) {
  // FIXME: Handle out-of-memory errors, returning -1 if OOM.
  std::lock_guard<std::recursive_mutex> Lock(JDStatesMutex);
  auto *JDS = getJITDylibStateByHeaderAddr(DSOHandle);
  assert(JDS && "JITDylib state not initialized");
  JDS->AtExits.push_back({F, Arg});
  return 0;
}

void MachOPlatformRuntimeState::runAtExits(void *DSOHandle) {
  // FIXME: Should atexits be allowed to run concurrently with access to
  // JDState?
  AtExitsVector V;
  {
    std::lock_guard<std::recursive_mutex> Lock(JDStatesMutex);
    auto *JDS = getJITDylibStateByHeaderAddr(DSOHandle);
    assert(JDS && "JITDlybi state not initialized");
    std::swap(V, JDS->AtExits);
  }

  while (!V.empty()) {
    auto &AE = V.back();
    AE.Func(AE.Arg);
    V.pop_back();
  }
}

Expected<std::pair<const char *, size_t>>
MachOPlatformRuntimeState::getThreadDataSectionFor(const char *ThreadData) {
  std::lock_guard<std::mutex> Lock(ThreadDataSectionsMutex);
  auto I = ThreadDataSections.upper_bound(ThreadData);
  // Check that we have a valid entry covering this address.
  if (I == ThreadDataSections.begin())
    return make_error<StringError>("No thread local data section for key");
  I = std::prev(I);
  if (ThreadData >= I->first + I->second)
    return make_error<StringError>("No thread local data section for key");
  return *I;
}

MachOPlatformRuntimeState::PerJITDylibState *
MachOPlatformRuntimeState::getJITDylibStateByHeaderAddr(void *DSOHandle) {
  auto I = JDStates.find(DSOHandle);
  if (I == JDStates.end())
    return nullptr;
  return &I->second;
}

MachOPlatformRuntimeState::PerJITDylibState *
MachOPlatformRuntimeState::getJITDylibStateByName(string_view Name) {
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

MachOPlatformRuntimeState::PerJITDylibState &
MachOPlatformRuntimeState::getOrCreateJITDylibState(
    MachOJITDylibInitializers &MOJDIs) {
  void *Header = MOJDIs.MachOHeaderAddress.toPtr<void *>();

  auto &JDS = JDStates[Header];

  // If this entry hasn't been created yet.
  if (!JDS.Header) {
    assert(!JDNameToHeader.count(MOJDIs.Name) &&
           "JITDylib has header map entry but no name map entry");
    JDNameToHeader[MOJDIs.Name] = Header;
    JDS.Header = Header;
  }

  return JDS;
}

Expected<ExecutorAddr>
MachOPlatformRuntimeState::lookupSymbolInJITDylib(void *DSOHandle,
                                                  string_view Sym) {
  Expected<ExecutorAddr> Result((ExecutorAddr()));
  if (auto Err = WrapperFunction<SPSExpected<SPSExecutorAddr>(
          SPSExecutorAddr, SPSString)>::call(&__orc_rt_macho_symbol_lookup_tag,
                                             Result,
                                             ExecutorAddr::fromPtr(DSOHandle),
                                             Sym))
    return std::move(Err);
  return Result;
}

Expected<MachOJITDylibInitializerSequence>
MachOPlatformRuntimeState::getJITDylibInitializersByName(string_view Path) {
  Expected<MachOJITDylibInitializerSequence> Result(
      (MachOJITDylibInitializerSequence()));
  std::string PathStr(Path.data(), Path.size());
  if (auto Err =
          WrapperFunction<SPSExpected<SPSMachOJITDylibInitializerSequence>(
              SPSString)>::call(&__orc_rt_macho_get_initializers_tag, Result,
                                Path))
    return std::move(Err);
  return Result;
}

Expected<void *> MachOPlatformRuntimeState::dlopenInitialize(string_view Path,
                                                             int Mode) {
  // Either our JITDylib wasn't loaded, or it or one of its dependencies allows
  // reinitialization. We need to call in to the JIT to see if there's any new
  // work pending.
  auto InitSeq = getJITDylibInitializersByName(Path);
  if (!InitSeq)
    return InitSeq.takeError();

  // Init sequences should be non-empty.
  if (InitSeq->empty())
    return make_error<StringError>(
        "__orc_rt_macho_get_initializers returned an "
        "empty init sequence");

  // Otherwise register and run initializers for each JITDylib.
  for (auto &MOJDIs : *InitSeq)
    if (auto Err = initializeJITDylib(MOJDIs))
      return std::move(Err);

  // Return the header for the last item in the list.
  auto *JDS = getJITDylibStateByHeaderAddr(
      InitSeq->back().MachOHeaderAddress.toPtr<void *>());
  assert(JDS && "Missing state entry for JD");
  return JDS->Header;
}

Error MachOPlatformRuntimeState::initializeJITDylib(
    MachOJITDylibInitializers &MOJDIs) {

  auto &JDS = getOrCreateJITDylibState(MOJDIs);
  ++JDS.RefCount;

  for (auto &KV : InitSections) {
    const auto &Name = KV.first;
    const auto &Handler = KV.second;
    auto I = MOJDIs.InitSections.find(Name);
    if (I != MOJDIs.InitSections.end()) {
      if (auto Err = Handler(I->second, MOJDIs))
        return Err;
    }
  }

  return Error::success();
}

class MachOPlatformRuntimeTLVManager {
public:
  void *getInstance(const char *ThreadData);

private:
  std::unordered_map<const char *, char *> Instances;
  std::unordered_map<const char *, std::unique_ptr<char[]>> AllocatedSections;
};

void *MachOPlatformRuntimeTLVManager::getInstance(const char *ThreadData) {
  auto I = Instances.find(ThreadData);
  if (I != Instances.end())
    return I->second;

  auto TDS =
      MachOPlatformRuntimeState::get().getThreadDataSectionFor(ThreadData);
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

void destroyMachOTLVMgr(void *MachOTLVMgr) {
  delete static_cast<MachOPlatformRuntimeTLVManager *>(MachOTLVMgr);
}

Error runWrapperFunctionCalls(std::vector<WrapperFunctionCall> WFCs) {
  for (auto &WFC : WFCs)
    if (auto Err = WFC.runWithSPSRet())
      return Err;
  return Error::success();
}

} // end anonymous namespace

//------------------------------------------------------------------------------
//                             JIT entry points
//------------------------------------------------------------------------------

ORC_RT_INTERFACE __orc_rt_CWrapperFunctionResult
__orc_rt_macho_platform_bootstrap(char *ArgData, size_t ArgSize) {
  MachOPlatformRuntimeState::initialize();
  return WrapperFunctionResult().release();
}

ORC_RT_INTERFACE __orc_rt_CWrapperFunctionResult
__orc_rt_macho_platform_shutdown(char *ArgData, size_t ArgSize) {
  MachOPlatformRuntimeState::destroy();
  return WrapperFunctionResult().release();
}

ORC_RT_INTERFACE __orc_rt_CWrapperFunctionResult
__orc_rt_macho_register_thread_data_section(char *ArgData, size_t ArgSize) {
  // NOTE: Does not use SPS to deserialize arg buffer, instead the arg buffer
  // is taken to be the range of the thread data section.
  return WrapperFunction<SPSError()>::handle(
             nullptr, 0,
             [&]() {
               return MachOPlatformRuntimeState::get()
                   .registerThreadDataSection(
                       span<const char>(ArgData, ArgSize));
             })
      .release();
}

ORC_RT_INTERFACE __orc_rt_CWrapperFunctionResult
__orc_rt_macho_deregister_thread_data_section(char *ArgData, size_t ArgSize) {
  // NOTE: Does not use SPS to deserialize arg buffer, instead the arg buffer
  // is taken to be the range of the thread data section.
  return WrapperFunction<SPSError()>::handle(
             nullptr, 0,
             [&]() {
               return MachOPlatformRuntimeState::get()
                   .deregisterThreadDataSection(
                       span<const char>(ArgData, ArgSize));
             })
      .release();
}

ORC_RT_INTERFACE __orc_rt_CWrapperFunctionResult
__orc_rt_macho_run_wrapper_function_calls(char *ArgData, size_t ArgSize) {
  return WrapperFunction<SPSError(SPSSequence<SPSWrapperFunctionCall>)>::handle(
             ArgData, ArgSize, runWrapperFunctionCalls)
      .release();
}

//------------------------------------------------------------------------------
//                            TLV support
//------------------------------------------------------------------------------

ORC_RT_INTERFACE void *__orc_rt_macho_tlv_get_addr_impl(TLVDescriptor *D) {
  auto *TLVMgr = static_cast<MachOPlatformRuntimeTLVManager *>(
      pthread_getspecific(D->Key));
  if (!TLVMgr) {
    TLVMgr = new MachOPlatformRuntimeTLVManager();
    if (pthread_setspecific(D->Key, TLVMgr)) {
      __orc_rt_log_error("Call to pthread_setspecific failed");
      return nullptr;
    }
  }

  return TLVMgr->getInstance(
      reinterpret_cast<char *>(static_cast<uintptr_t>(D->DataAddress)));
}

ORC_RT_INTERFACE __orc_rt_CWrapperFunctionResult
__orc_rt_macho_create_pthread_key(char *ArgData, size_t ArgSize) {
  return WrapperFunction<SPSExpected<uint64_t>(void)>::handle(
             ArgData, ArgSize,
             []() -> Expected<uint64_t> {
               pthread_key_t Key;
               if (int Err = pthread_key_create(&Key, destroyMachOTLVMgr)) {
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

int __orc_rt_macho_cxa_atexit(void (*func)(void *), void *arg,
                              void *dso_handle) {
  return MachOPlatformRuntimeState::get().registerAtExit(func, arg, dso_handle);
}

void __orc_rt_macho_cxa_finalize(void *dso_handle) {
  MachOPlatformRuntimeState::get().runAtExits(dso_handle);
}

//------------------------------------------------------------------------------
//                        JIT'd dlfcn alternatives.
//------------------------------------------------------------------------------

const char *__orc_rt_macho_jit_dlerror() {
  return MachOPlatformRuntimeState::get().dlerror();
}

void *__orc_rt_macho_jit_dlopen(const char *path, int mode) {
  return MachOPlatformRuntimeState::get().dlopen(path, mode);
}

int __orc_rt_macho_jit_dlclose(void *dso_handle) {
  return MachOPlatformRuntimeState::get().dlclose(dso_handle);
}

void *__orc_rt_macho_jit_dlsym(void *dso_handle, const char *symbol) {
  return MachOPlatformRuntimeState::get().dlsym(dso_handle, symbol);
}

//------------------------------------------------------------------------------
//                             MachO Run Program
//------------------------------------------------------------------------------

ORC_RT_INTERFACE int64_t __orc_rt_macho_run_program(const char *JITDylibName,
                                                    const char *EntrySymbolName,
                                                    int argc, char *argv[]) {
  using MainTy = int (*)(int, char *[]);

  void *H = __orc_rt_macho_jit_dlopen(JITDylibName,
                                      __orc_rt::macho::ORC_RT_RTLD_LAZY);
  if (!H) {
    __orc_rt_log_error(__orc_rt_macho_jit_dlerror());
    return -1;
  }

  auto *Main =
      reinterpret_cast<MainTy>(__orc_rt_macho_jit_dlsym(H, EntrySymbolName));

  if (!Main) {
    __orc_rt_log_error(__orc_rt_macho_jit_dlerror());
    return -1;
  }

  int Result = Main(argc, argv);

  if (__orc_rt_macho_jit_dlclose(H) == -1)
    __orc_rt_log_error(__orc_rt_macho_jit_dlerror());

  return Result;
}
