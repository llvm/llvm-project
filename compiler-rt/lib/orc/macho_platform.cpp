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
#include "debug.h"
#include "error.h"
#include "wrapper_function_utils.h"

#include <algorithm>
#include <ios>
#include <map>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define DEBUG_TYPE "macho_platform"

using namespace __orc_rt;
using namespace __orc_rt::macho;

// Declare function tags for functions in the JIT process.
ORC_RT_JIT_DISPATCH_TAG(__orc_rt_macho_push_initializers_tag)
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

struct MachOJITDylibDepInfo {
  bool Sealed = false;
  std::vector<ExecutorAddr> DepHeaders;
};

using MachOJITDylibDepInfoMap =
    std::unordered_map<ExecutorAddr, MachOJITDylibDepInfo>;

} // anonymous namespace

namespace __orc_rt {

using SPSMachOObjectPlatformSectionsMap =
    SPSSequence<SPSTuple<SPSString, SPSExecutorAddrRange>>;

using SPSMachOJITDylibDepInfo = SPSTuple<bool, SPSSequence<SPSExecutorAddr>>;

using SPSMachOJITDylibDepInfoMap =
    SPSSequence<SPSTuple<SPSExecutorAddr, SPSMachOJITDylibDepInfo>>;

template <>
class SPSSerializationTraits<SPSMachOJITDylibDepInfo, MachOJITDylibDepInfo> {
public:
  static size_t size(const MachOJITDylibDepInfo &JDI) {
    return SPSMachOJITDylibDepInfo::AsArgList::size(JDI.Sealed, JDI.DepHeaders);
  }

  static bool serialize(SPSOutputBuffer &OB, const MachOJITDylibDepInfo &JDI) {
    return SPSMachOJITDylibDepInfo::AsArgList::serialize(OB, JDI.Sealed,
                                                         JDI.DepHeaders);
  }

  static bool deserialize(SPSInputBuffer &IB, MachOJITDylibDepInfo &JDI) {
    return SPSMachOJITDylibDepInfo::AsArgList::deserialize(IB, JDI.Sealed,
                                                           JDI.DepHeaders);
  }
};

} // namespace __orc_rt

namespace {
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

  struct JITDylibState {
    std::string Name;
    void *Header = nullptr;
    bool Sealed = false;
    size_t LinkedAgainstRefCount = 0;
    size_t DlRefCount = 0;
    std::vector<JITDylibState *> Deps;
    AtExitsVector AtExits;
    const objc_image_info *ObjCImageInfo = nullptr;
    std::vector<span<void (*)()>> ModInitsSections;
    std::vector<span<void (*)()>> ModInitsSectionsNew;
    std::vector<span<uintptr_t>> ObjCClassListSections;
    std::vector<span<uintptr_t>> ObjCClassListSectionsNew;
    std::vector<span<uintptr_t>> ObjCSelRefsSections;
    std::vector<span<uintptr_t>> ObjCSelRefsSectionsNew;
    std::vector<span<char>> Swift5ProtoSections;
    std::vector<span<char>> Swift5ProtoSectionsNew;
    std::vector<span<char>> Swift5ProtosSections;
    std::vector<span<char>> Swift5ProtosSectionsNew;
    std::vector<span<char>> Swift5TypesSections;
    std::vector<span<char>> Swift5TypesSectionsNew;

    bool referenced() const {
      return LinkedAgainstRefCount != 0 || DlRefCount != 0;
    }
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

  Error registerJITDylib(std::string Name, void *Header);
  Error deregisterJITDylib(void *Header);
  Error registerThreadDataSection(span<const char> ThreadDataSection);
  Error deregisterThreadDataSection(span<const char> ThreadDataSection);
  Error registerObjectPlatformSections(
      ExecutorAddr HeaderAddr,
      std::vector<std::pair<string_view, ExecutorAddrRange>> Secs);
  Error deregisterObjectPlatformSections(
      ExecutorAddr HeaderAddr,
      std::vector<std::pair<string_view, ExecutorAddrRange>> Secs);

  const char *dlerror();
  void *dlopen(string_view Name, int Mode);
  int dlclose(void *DSOHandle);
  void *dlsym(void *DSOHandle, string_view Symbol);

  int registerAtExit(void (*F)(void *), void *Arg, void *DSOHandle);
  void runAtExits(JITDylibState &JDS);
  void runAtExits(void *DSOHandle);

  /// Returns the base address of the section containing ThreadData.
  Expected<std::pair<const char *, size_t>>
  getThreadDataSectionFor(const char *ThreadData);

private:
  JITDylibState *getJITDylibStateByHeader(void *DSOHandle);
  JITDylibState *getJITDylibStateByName(string_view Path);

  Expected<ExecutorAddr> lookupSymbolInJITDylib(void *DSOHandle,
                                                string_view Symbol);

  static Error registerObjCSelectors(JITDylibState &JDS);
  static Error registerObjCClasses(JITDylibState &JDS);
  static Error registerSwift5Protocols(JITDylibState &JDS);
  static Error registerSwift5ProtocolConformances(JITDylibState &JDS);
  static Error registerSwift5Types(JITDylibState &JDS);
  static Error runModInits(JITDylibState &JDS);

  Expected<void *> dlopenImpl(string_view Path, int Mode);
  Error dlopenFull(JITDylibState &JDS);
  Error dlopenInitialize(JITDylibState &JDS, MachOJITDylibDepInfoMap &DepInfo);

  Error dlcloseImpl(void *DSOHandle);
  Error dlcloseDeinitialize(JITDylibState &JDS);

  static MachOPlatformRuntimeState *MOPS;

  // FIXME: Move to thread-state.
  std::string DLFcnError;

  std::recursive_mutex JDStatesMutex;
  std::unordered_map<void *, JITDylibState> JDStates;
  std::unordered_map<string_view, void *> JDNameToHeader;

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

Error MachOPlatformRuntimeState::registerJITDylib(std::string Name,
                                                  void *Header) {
  ORC_RT_DEBUG({
    printdbg("Registering JITDylib %s: Header = %p\n", Name.c_str(), Header);
  });
  std::lock_guard<std::recursive_mutex> Lock(JDStatesMutex);
  if (JDStates.count(Header)) {
    std::ostringstream ErrStream;
    ErrStream << "Duplicate JITDylib registration for header " << Header
              << " (name = " << Name << ")";
    return make_error<StringError>(ErrStream.str());
  }
  if (JDNameToHeader.count(Name)) {
    std::ostringstream ErrStream;
    ErrStream << "Duplicate JITDylib registration for header " << Header
              << " (header = " << Header << ")";
    return make_error<StringError>(ErrStream.str());
  }

  auto &JDS = JDStates[Header];
  JDS.Name = std::move(Name);
  JDS.Header = Header;
  JDNameToHeader[JDS.Name] = Header;
  return Error::success();
}

Error MachOPlatformRuntimeState::deregisterJITDylib(void *Header) {
  std::lock_guard<std::recursive_mutex> Lock(JDStatesMutex);
  auto I = JDStates.find(Header);
  if (I == JDStates.end()) {
    std::ostringstream ErrStream;
    ErrStream << "Attempted to deregister unrecognized header " << Header;
    return make_error<StringError>(ErrStream.str());
  }

  // Remove std::string construction once we can use C++20.
  auto J = JDNameToHeader.find(
      std::string(I->second.Name.data(), I->second.Name.size()));
  assert(J != JDNameToHeader.end() &&
         "Missing JDNameToHeader entry for JITDylib");

  ORC_RT_DEBUG({
    printdbg("Deregistering JITDylib %s: Header = %p\n", I->second.Name.c_str(),
             Header);
  });

  JDNameToHeader.erase(J);
  JDStates.erase(I);
  return Error::success();
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

Error MachOPlatformRuntimeState::registerObjectPlatformSections(
    ExecutorAddr HeaderAddr,
    std::vector<std::pair<string_view, ExecutorAddrRange>> Secs) {
  ORC_RT_DEBUG({
    printdbg("MachOPlatform: Registering object sections for %p.\n",
             HeaderAddr.toPtr<void *>());
  });

  std::lock_guard<std::recursive_mutex> Lock(JDStatesMutex);
  auto *JDS = getJITDylibStateByHeader(HeaderAddr.toPtr<void *>());
  if (!JDS) {
    std::ostringstream ErrStream;
    ErrStream << "Could not register object platform sections for "
                 "unrecognized header "
              << HeaderAddr.toPtr<void *>();
    return make_error<StringError>(ErrStream.str());
  }

  for (auto &KV : Secs) {
    // FIXME: Validate section ranges?
    if (KV.first == "__DATA,__thread_data") {
      if (auto Err = registerThreadDataSection(KV.second.toSpan<const char>()))
        return Err;
    } else if (KV.first == "__DATA,__objc_selrefs")
      JDS->ObjCSelRefsSectionsNew.push_back(KV.second.toSpan<uintptr_t>());
    else if (KV.first == "__DATA,__objc_classlist")
      JDS->ObjCClassListSectionsNew.push_back(KV.second.toSpan<uintptr_t>());
    else if (KV.first == "__TEXT,__swift5_protos")
      JDS->Swift5ProtosSectionsNew.push_back(KV.second.toSpan<char>());
    else if (KV.first == "__TEXT,__swift5_proto")
      JDS->Swift5ProtoSectionsNew.push_back(KV.second.toSpan<char>());
    else if (KV.first == "__TEXT,__swift5_types")
      JDS->Swift5TypesSectionsNew.push_back(KV.second.toSpan<char>());
    else if (KV.first == "__DATA,__mod_init_func")
      JDS->ModInitsSectionsNew.push_back(KV.second.toSpan<void (*)()>());
    else {
      // Should this be a warning instead?
      return make_error<StringError>(
          "Encountered unexpected section " +
          std::string(KV.first.data(), KV.first.size()) +
          " while registering object platform sections");
    }
  }

  return Error::success();
}

// Remove the given range from the given vector if present.
// Returns true if the range was removed, false otherwise.
template <typename T>
bool removeIfPresent(std::vector<span<T>> &V, ExecutorAddrRange R) {
  auto RI = std::find_if(
      V.rbegin(), V.rend(),
      [RS = R.toSpan<T>()](const span<T> &E) { return E.data() == RS.data(); });
  if (RI != V.rend()) {
    V.erase(std::next(RI).base());
    return true;
  }
  return false;
}

Error MachOPlatformRuntimeState::deregisterObjectPlatformSections(
    ExecutorAddr HeaderAddr,
    std::vector<std::pair<string_view, ExecutorAddrRange>> Secs) {
  // TODO: Make this more efficient? (maybe unnecessary if removal is rare?)
  // TODO: Add a JITDylib prepare-for-teardown operation that clears all
  //       registered sections, causing this function to take the fast-path.
  ORC_RT_DEBUG({
    printdbg("MachOPlatform: Registering object sections for %p.\n",
             HeaderAddr.toPtr<void *>());
  });

  std::lock_guard<std::recursive_mutex> Lock(JDStatesMutex);
  auto *JDS = getJITDylibStateByHeader(HeaderAddr.toPtr<void *>());
  if (!JDS) {
    std::ostringstream ErrStream;
    ErrStream << "Could not register object platform sections for unrecognized "
                 "header "
              << HeaderAddr.toPtr<void *>();
    return make_error<StringError>(ErrStream.str());
  }

  // FIXME: Implement faster-path by returning immediately if JDS is being
  // torn down entirely?

  for (auto &KV : Secs) {
    // FIXME: Validate section ranges?
    if (KV.first == "__DATA,__thread_data") {
      if (auto Err =
              deregisterThreadDataSection(KV.second.toSpan<const char>()))
        return Err;
    } else if (KV.first == "__DATA,__objc_selrefs") {
      if (!removeIfPresent(JDS->ObjCSelRefsSections, KV.second))
        removeIfPresent(JDS->ObjCSelRefsSectionsNew, KV.second);
    } else if (KV.first == "__DATA,__objc_classlist") {
      if (!removeIfPresent(JDS->ObjCClassListSections, KV.second))
        removeIfPresent(JDS->ObjCClassListSectionsNew, KV.second);
    } else if (KV.first == "__TEXT,__swift5_protos") {
      if (!removeIfPresent(JDS->Swift5ProtosSections, KV.second))
        removeIfPresent(JDS->Swift5ProtosSectionsNew, KV.second);
    } else if (KV.first == "__TEXT,__swift5_proto") {
      if (!removeIfPresent(JDS->Swift5ProtoSections, KV.second))
        removeIfPresent(JDS->Swift5ProtoSectionsNew, KV.second);
    } else if (KV.first == "__TEXT,__swift5_types") {
      if (!removeIfPresent(JDS->Swift5TypesSections, KV.second))
        removeIfPresent(JDS->Swift5TypesSectionsNew, KV.second);
    } else if (KV.first == "__DATA,__mod_init_func") {
      if (!removeIfPresent(JDS->ModInitsSections, KV.second))
        removeIfPresent(JDS->ModInitsSectionsNew, KV.second);
    } else {
      // Should this be a warning instead?
      return make_error<StringError>(
          "Encountered unexpected section " +
          std::string(KV.first.data(), KV.first.size()) +
          " while deregistering object platform sections");
    }
  }
  return Error::success();
}

const char *MachOPlatformRuntimeState::dlerror() { return DLFcnError.c_str(); }

void *MachOPlatformRuntimeState::dlopen(string_view Path, int Mode) {
  ORC_RT_DEBUG({
    std::string S(Path.data(), Path.size());
    printdbg("MachOPlatform::dlopen(\"%s\")\n", S.c_str());
  });
  std::lock_guard<std::recursive_mutex> Lock(JDStatesMutex);
  if (auto H = dlopenImpl(Path, Mode))
    return *H;
  else {
    // FIXME: Make dlerror thread safe.
    DLFcnError = toString(H.takeError());
    return nullptr;
  }
}

int MachOPlatformRuntimeState::dlclose(void *DSOHandle) {
  ORC_RT_DEBUG({
    auto *JDS = getJITDylibStateByHeader(DSOHandle);
    std::string DylibName;
    if (JDS) {
      std::string S;
      printdbg("MachOPlatform::dlclose(%p) (%s)\n", DSOHandle, S.c_str());
    } else
      printdbg("MachOPlatform::dlclose(%p) (%s)\n", DSOHandle,
               "invalid handle");
  });
  std::lock_guard<std::recursive_mutex> Lock(JDStatesMutex);
  if (auto Err = dlcloseImpl(DSOHandle)) {
    // FIXME: Make dlerror thread safe.
    DLFcnError = toString(std::move(Err));
    return -1;
  }
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
  auto *JDS = getJITDylibStateByHeader(DSOHandle);
  if (!JDS) {
    ORC_RT_DEBUG({
      printdbg("MachOPlatformRuntimeState::registerAtExit called with "
               "unrecognized dso handle %p\n",
               DSOHandle);
    });
    return -1;
  }
  JDS->AtExits.push_back({F, Arg});
  return 0;
}

void MachOPlatformRuntimeState::runAtExits(JITDylibState &JDS) {
  while (!JDS.AtExits.empty()) {
    auto &AE = JDS.AtExits.back();
    AE.Func(AE.Arg);
    JDS.AtExits.pop_back();
  }
}

void MachOPlatformRuntimeState::runAtExits(void *DSOHandle) {
  std::lock_guard<std::recursive_mutex> Lock(JDStatesMutex);
  auto *JDS = getJITDylibStateByHeader(DSOHandle);
  ORC_RT_DEBUG({
    printdbg("MachOPlatformRuntimeState::runAtExits called on unrecognized "
             "dso_handle %p\n",
             DSOHandle);
  });
  if (JDS)
    runAtExits(*JDS);
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

MachOPlatformRuntimeState::JITDylibState *
MachOPlatformRuntimeState::getJITDylibStateByHeader(void *DSOHandle) {
  auto I = JDStates.find(DSOHandle);
  if (I == JDStates.end()) {
    I = JDStates.insert(std::make_pair(DSOHandle, JITDylibState())).first;
    I->second.Header = DSOHandle;
  }
  return &I->second;
}

MachOPlatformRuntimeState::JITDylibState *
MachOPlatformRuntimeState::getJITDylibStateByName(string_view Name) {
  // FIXME: Avoid creating string once we have C++20.
  auto I = JDNameToHeader.find(std::string(Name.data(), Name.size()));
  if (I != JDNameToHeader.end())
    return getJITDylibStateByHeader(I->second);
  return nullptr;
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

template <typename T>
static void moveAppendSections(std::vector<span<T>> &Dst,
                               std::vector<span<T>> &Src) {
  if (Dst.empty()) {
    Dst = std::move(Src);
    return;
  }

  Dst.reserve(Dst.size() + Src.size());
  std::copy(Src.begin(), Src.end(), std::back_inserter(Dst));
  Src.clear();
}

Error MachOPlatformRuntimeState::registerObjCSelectors(JITDylibState &JDS) {

  if (JDS.ObjCSelRefsSectionsNew.empty())
    return Error::success();

  if (ORC_RT_UNLIKELY(!sel_registerName))
    return make_error<StringError>("sel_registerName is not available");

  for (const auto &ObjCSelRefs : JDS.ObjCSelRefsSectionsNew) {
    for (uintptr_t &SelEntry : ObjCSelRefs) {
      const char *SelName = reinterpret_cast<const char *>(SelEntry);
      auto Sel = sel_registerName(SelName);
      *reinterpret_cast<SEL *>(&SelEntry) = Sel;
    }
  }

  moveAppendSections(JDS.ObjCSelRefsSections, JDS.ObjCSelRefsSectionsNew);
  return Error::success();
}

Error MachOPlatformRuntimeState::registerObjCClasses(JITDylibState &JDS) {

  if (JDS.ObjCClassListSectionsNew.empty())
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

  auto ClassSelector = sel_registerName("class");

  for (const auto &ObjCClassList : JDS.ObjCClassListSectionsNew) {
    for (uintptr_t ClassPtr : ObjCClassList) {
      auto *Cls = reinterpret_cast<Class>(ClassPtr);
      auto *ClassCompiled = reinterpret_cast<ObjCClassCompiled *>(ClassPtr);
      objc_msgSend(reinterpret_cast<id>(ClassCompiled->Parent), ClassSelector);
      auto Registered = objc_readClassPair(Cls, JDS.ObjCImageInfo);

      // FIXME: Improve diagnostic by reporting the failed class's name.
      if (Registered != Cls)
        return make_error<StringError>("Unable to register Objective-C class");
    }
  }

  moveAppendSections(JDS.ObjCClassListSections, JDS.ObjCClassListSectionsNew);
  return Error::success();
}

Error MachOPlatformRuntimeState::registerSwift5Protocols(JITDylibState &JDS) {

  if (JDS.Swift5ProtosSectionsNew.empty())
    return Error::success();

  if (ORC_RT_UNLIKELY(!swift_registerProtocols))
    return make_error<StringError>("swift_registerProtocols is not available");

  for (const auto &Swift5Protocols : JDS.Swift5ProtoSectionsNew)
    swift_registerProtocols(
        reinterpret_cast<const ProtocolRecord *>(Swift5Protocols.data()),
        reinterpret_cast<const ProtocolRecord *>(Swift5Protocols.data() +
                                                 Swift5Protocols.size()));

  moveAppendSections(JDS.Swift5ProtoSections, JDS.Swift5ProtoSectionsNew);
  return Error::success();
}

Error MachOPlatformRuntimeState::registerSwift5ProtocolConformances(
    JITDylibState &JDS) {

  if (JDS.Swift5ProtosSectionsNew.empty())
    return Error::success();

  if (ORC_RT_UNLIKELY(!swift_registerProtocolConformances))
    return make_error<StringError>(
        "swift_registerProtocolConformances is not available");

  for (const auto &ProtoConfSec : JDS.Swift5ProtosSectionsNew)
    swift_registerProtocolConformances(
        reinterpret_cast<const ProtocolConformanceRecord *>(
            ProtoConfSec.data()),
        reinterpret_cast<const ProtocolConformanceRecord *>(
            ProtoConfSec.data() + ProtoConfSec.size()));

  moveAppendSections(JDS.Swift5ProtosSections, JDS.Swift5ProtosSectionsNew);
  return Error::success();
}

Error MachOPlatformRuntimeState::registerSwift5Types(JITDylibState &JDS) {

  if (JDS.Swift5TypesSectionsNew.empty())
    return Error::success();

  if (ORC_RT_UNLIKELY(!swift_registerTypeMetadataRecords))
    return make_error<StringError>(
        "swift_registerTypeMetadataRecords is not available");

  for (const auto &TypeSec : JDS.Swift5TypesSectionsNew)
    swift_registerTypeMetadataRecords(
        reinterpret_cast<const TypeMetadataRecord *>(TypeSec.data()),
        reinterpret_cast<const TypeMetadataRecord *>(TypeSec.data() +
                                                     TypeSec.size()));

  moveAppendSections(JDS.Swift5TypesSections, JDS.Swift5TypesSectionsNew);
  return Error::success();
}

Error MachOPlatformRuntimeState::runModInits(JITDylibState &JDS) {

  for (const auto &ModInits : JDS.ModInitsSectionsNew) {
    for (void (*Init)() : ModInits)
      (*Init)();
  }

  moveAppendSections(JDS.ModInitsSections, JDS.ModInitsSectionsNew);
  return Error::success();
}

Expected<void *> MachOPlatformRuntimeState::dlopenImpl(string_view Path,
                                                       int Mode) {
  // Try to find JITDylib state by name.
  auto *JDS = getJITDylibStateByName(Path);

  if (!JDS)
    return make_error<StringError>("No registered JTIDylib for path " +
                                   std::string(Path.data(), Path.size()));

  // If this JITDylib is unsealed, or this is the first dlopen then run
  // full dlopen path (update deps, push and run initializers, update ref
  // counts on all JITDylibs in the dep tree).
  if (!JDS->referenced() || !JDS->Sealed) {
    if (auto Err = dlopenFull(*JDS))
      return std::move(Err);
  }

  // Bump the ref-count on this dylib.
  ++JDS->DlRefCount;

  // Return the header address.
  return JDS->Header;
}

Error MachOPlatformRuntimeState::dlopenFull(JITDylibState &JDS) {
  // Call back to the JIT to push the initializers.
  Expected<MachOJITDylibDepInfoMap> DepInfo((MachOJITDylibDepInfoMap()));
  if (auto Err = WrapperFunction<SPSExpected<SPSMachOJITDylibDepInfoMap>(
          SPSExecutorAddr)>::call(&__orc_rt_macho_push_initializers_tag,
                                  DepInfo, ExecutorAddr::fromPtr(JDS.Header)))
    return Err;
  if (!DepInfo)
    return DepInfo.takeError();

  if (auto Err = dlopenInitialize(JDS, *DepInfo))
    return Err;

  if (!DepInfo->empty()) {
    ORC_RT_DEBUG({
      printdbg("Unrecognized dep-info key headers in dlopen of %s\n",
               JDS.Name.c_str());
    });
    std::ostringstream ErrStream;
    ErrStream << "Encountered unrecognized dep-info key headers "
                 "while processing dlopen of "
              << JDS.Name;
    return make_error<StringError>(ErrStream.str());
  }

  return Error::success();
}

Error MachOPlatformRuntimeState::dlopenInitialize(
    JITDylibState &JDS, MachOJITDylibDepInfoMap &DepInfo) {
  ORC_RT_DEBUG({
    printdbg("MachOPlatformRuntimeState::dlopenInitialize(\"%s\")\n",
             JDS.Name.c_str());
  });

  // If the header is not present in the dep map then assume that we
  // already processed it earlier in the dlopenInitialize traversal and
  // return.
  // TODO: Keep a visited set instead so that we can error out on missing
  //       entries?
  auto I = DepInfo.find(ExecutorAddr::fromPtr(JDS.Header));
  if (I == DepInfo.end())
    return Error::success();

  auto DI = std::move(I->second);
  DepInfo.erase(I);

  // We don't need to re-initialize sealed JITDylibs that have already been
  // initialized. Just check that their dep-map entry is empty as expected.
  if (JDS.Sealed) {
    if (!DI.DepHeaders.empty()) {
      std::ostringstream ErrStream;
      ErrStream << "Sealed JITDylib " << JDS.Header
                << " already has registered dependencies";
      return make_error<StringError>(ErrStream.str());
    }
    if (JDS.referenced())
      return Error::success();
  } else
    JDS.Sealed = DI.Sealed;

  // This is an unsealed or newly sealed JITDylib. Run initializers.
  std::vector<JITDylibState *> OldDeps;
  std::swap(JDS.Deps, OldDeps);
  JDS.Deps.reserve(DI.DepHeaders.size());
  for (auto DepHeaderAddr : DI.DepHeaders) {
    auto *DepJDS = getJITDylibStateByHeader(DepHeaderAddr.toPtr<void *>());
    if (!DepJDS) {
      std::ostringstream ErrStream;
      ErrStream << "Encountered unrecognized dep header "
                << DepHeaderAddr.toPtr<void *>() << " while initializing "
                << JDS.Name;
      return make_error<StringError>(ErrStream.str());
    }
    ++DepJDS->LinkedAgainstRefCount;
    if (auto Err = dlopenInitialize(*DepJDS, DepInfo))
      return Err;
  }

  // Initialize this JITDylib.
  if (auto Err = registerObjCSelectors(JDS))
    return Err;
  if (auto Err = registerObjCClasses(JDS))
    return Err;
  if (auto Err = registerSwift5Protocols(JDS))
    return Err;
  if (auto Err = registerSwift5ProtocolConformances(JDS))
    return Err;
  if (auto Err = registerSwift5Types(JDS))
    return Err;
  if (auto Err = runModInits(JDS))
    return Err;

  // Decrement old deps.
  // FIXME: We should probably continue and just report deinitialize errors
  // here.
  for (auto *DepJDS : OldDeps) {
    --DepJDS->LinkedAgainstRefCount;
    if (!DepJDS->referenced())
      if (auto Err = dlcloseDeinitialize(*DepJDS))
        return Err;
  }

  return Error::success();
}

Error MachOPlatformRuntimeState::dlcloseImpl(void *DSOHandle) {
  // Try to find JITDylib state by header.
  auto *JDS = getJITDylibStateByHeader(DSOHandle);

  if (!JDS) {
    std::ostringstream ErrStream;
    ErrStream << "No registered JITDylib for " << DSOHandle;
    return make_error<StringError>(ErrStream.str());
  }

  // Bump the ref-count.
  --JDS->DlRefCount;

  if (!JDS->referenced())
    return dlcloseDeinitialize(*JDS);

  return Error::success();
}

Error MachOPlatformRuntimeState::dlcloseDeinitialize(JITDylibState &JDS) {

  ORC_RT_DEBUG({
    printdbg("MachOPlatformRuntimeState::dlcloseDeinitialize(\"%s\")\n",
             JDS.Name.c_str());
  });

  runAtExits(JDS);

  // Reset mod-inits
  moveAppendSections(JDS.ModInitsSections, JDS.ModInitsSectionsNew);
  JDS.ModInitsSectionsNew = std::move(JDS.ModInitsSections);

  // Deinitialize any dependencies.
  for (auto *DepJDS : JDS.Deps) {
    --DepJDS->LinkedAgainstRefCount;
    if (!DepJDS->referenced())
      if (auto Err = dlcloseDeinitialize(*DepJDS))
        return Err;
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
    if (auto Err = WFC.runWithSPSRet<void>())
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
__orc_rt_macho_register_jitdylib(char *ArgData, size_t ArgSize) {
  return WrapperFunction<SPSError(SPSString, SPSExecutorAddr)>::handle(
             ArgData, ArgSize,
             [](std::string &Name, ExecutorAddr HeaderAddr) {
               return MachOPlatformRuntimeState::get().registerJITDylib(
                   std::move(Name), HeaderAddr.toPtr<void *>());
             })
      .release();
}

ORC_RT_INTERFACE __orc_rt_CWrapperFunctionResult
__orc_rt_macho_deregister_jitdylib(char *ArgData, size_t ArgSize) {
  return WrapperFunction<SPSError(SPSExecutorAddr)>::handle(
             ArgData, ArgSize,
             [](ExecutorAddr HeaderAddr) {
               return MachOPlatformRuntimeState::get().deregisterJITDylib(
                   HeaderAddr.toPtr<void *>());
             })
      .release();
}

ORC_RT_INTERFACE __orc_rt_CWrapperFunctionResult
__orc_rt_macho_register_object_platform_sections(char *ArgData,
                                                 size_t ArgSize) {
  return WrapperFunction<SPSError(SPSExecutorAddr,
                                  SPSMachOObjectPlatformSectionsMap)>::
      handle(ArgData, ArgSize,
             [](ExecutorAddr HeaderAddr,
                std::vector<std::pair<string_view, ExecutorAddrRange>> &Secs) {
               return MachOPlatformRuntimeState::get()
                   .registerObjectPlatformSections(HeaderAddr, std::move(Secs));
             })
          .release();
}

ORC_RT_INTERFACE __orc_rt_CWrapperFunctionResult
__orc_rt_macho_deregister_object_platform_sections(char *ArgData,
                                                   size_t ArgSize) {
  return WrapperFunction<SPSError(SPSExecutorAddr,
                                  SPSMachOObjectPlatformSectionsMap)>::
      handle(ArgData, ArgSize,
             [](ExecutorAddr HeaderAddr,
                std::vector<std::pair<string_view, ExecutorAddrRange>> &Secs) {
               return MachOPlatformRuntimeState::get()
                   .deregisterObjectPlatformSections(HeaderAddr,
                                                     std::move(Secs));
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
