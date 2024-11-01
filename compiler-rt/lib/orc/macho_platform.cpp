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
#include <string_view>
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

  /// Used to manage sections of fixed-sized metadata records (e.g. pointer
  /// sections, selector refs, etc.)
  template <typename RecordElement> class RecordSectionsTracker {
  public:
    /// Add a section to the "new" list.
    void add(span<RecordElement> Sec) { New.push_back(std::move(Sec)); }

    /// Returns true if there are new sections to process.
    bool hasNewSections() const { return !New.empty(); }

    /// Returns the number of new sections to process.
    size_t numNewSections() const { return New.size(); }

    /// Process all new sections.
    template <typename ProcessSectionFunc>
    std::enable_if_t<std::is_void_v<
        std::invoke_result_t<ProcessSectionFunc, span<RecordElement>>>>
    processNewSections(ProcessSectionFunc &&ProcessSection) {
      for (auto &Sec : New)
        ProcessSection(Sec);
      moveNewToProcessed();
    }

    /// Proces all new sections with a fallible handler.
    ///
    /// Successfully handled sections will be moved to the Processed
    /// list.
    template <typename ProcessSectionFunc>
    std::enable_if_t<
        std::is_same_v<Error, std::invoke_result_t<ProcessSectionFunc,
                                                   span<RecordElement>>>,
        Error>
    processNewSections(ProcessSectionFunc &&ProcessSection) {
      for (size_t I = 0; I != New.size(); ++I) {
        if (auto Err = ProcessSection(New[I])) {
          for (size_t J = 0; J != I; ++J)
            Processed.push_back(New[J]);
          New.erase(New.begin(), New.begin() + I);
          return Err;
        }
      }
      moveNewToProcessed();
      return Error::success();
    }

    /// Move all sections back to New for reprocessing.
    void reset() {
      moveNewToProcessed();
      New = std::move(Processed);
    }

    /// Remove the section with the given range.
    bool removeIfPresent(ExecutorAddrRange R) {
      if (removeIfPresent(New, R))
        return true;
      return removeIfPresent(Processed, R);
    }

  private:
    void moveNewToProcessed() {
      if (Processed.empty())
        Processed = std::move(New);
      else {
        Processed.reserve(Processed.size() + New.size());
        std::copy(New.begin(), New.end(), std::back_inserter(Processed));
        New.clear();
      }
    }

    bool removeIfPresent(std::vector<span<RecordElement>> &V,
                         ExecutorAddrRange R) {
      auto RI = std::find_if(
          V.rbegin(), V.rend(),
          [RS = R.toSpan<RecordElement>()](const span<RecordElement> &E) {
            return E.data() == RS.data();
          });
      if (RI != V.rend()) {
        V.erase(std::next(RI).base());
        return true;
      }
      return false;
    }

    std::vector<span<RecordElement>> Processed;
    std::vector<span<RecordElement>> New;
  };

  struct JITDylibState {
    std::string Name;
    void *Header = nullptr;
    bool Sealed = false;
    size_t LinkedAgainstRefCount = 0;
    size_t DlRefCount = 0;
    std::vector<JITDylibState *> Deps;
    AtExitsVector AtExits;
    const objc_image_info *ObjCImageInfo = nullptr;
    std::unordered_map<void *, std::vector<char>> DataSectionContent;
    std::unordered_map<void *, size_t> ZeroInitRanges;
    RecordSectionsTracker<void (*)()> ModInitsSections;
    RecordSectionsTracker<void *> ObjCClassListSections;
    RecordSectionsTracker<void *> ObjCSelRefsSections;
    RecordSectionsTracker<char> Swift5ProtocolsSections;
    RecordSectionsTracker<char> Swift5ProtocolConformancesSections;
    RecordSectionsTracker<char> Swift5TypesSections;

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
      std::vector<std::pair<std::string_view, ExecutorAddrRange>> Secs);
  Error deregisterObjectPlatformSections(
      ExecutorAddr HeaderAddr,
      std::vector<std::pair<std::string_view, ExecutorAddrRange>> Secs);

  const char *dlerror();
  void *dlopen(std::string_view Name, int Mode);
  int dlclose(void *DSOHandle);
  void *dlsym(void *DSOHandle, std::string_view Symbol);

  int registerAtExit(void (*F)(void *), void *Arg, void *DSOHandle);
  void runAtExits(std::unique_lock<std::mutex> &JDStatesLock,
                  JITDylibState &JDS);
  void runAtExits(void *DSOHandle);

  /// Returns the base address of the section containing ThreadData.
  Expected<std::pair<const char *, size_t>>
  getThreadDataSectionFor(const char *ThreadData);

private:
  JITDylibState *getJITDylibStateByHeader(void *DSOHandle);
  JITDylibState *getJITDylibStateByName(std::string_view Path);

  Expected<ExecutorAddr> lookupSymbolInJITDylib(void *DSOHandle,
                                                std::string_view Symbol);

  static Error registerObjCSelectors(JITDylibState &JDS);
  static Error registerObjCClasses(JITDylibState &JDS);
  static Error registerSwift5Protocols(JITDylibState &JDS);
  static Error registerSwift5ProtocolConformances(JITDylibState &JDS);
  static Error registerSwift5Types(JITDylibState &JDS);
  static Error runModInits(std::unique_lock<std::mutex> &JDStatesLock,
                           JITDylibState &JDS);

  Expected<void *> dlopenImpl(std::string_view Path, int Mode);
  Error dlopenFull(std::unique_lock<std::mutex> &JDStatesLock,
                   JITDylibState &JDS);
  Error dlopenInitialize(std::unique_lock<std::mutex> &JDStatesLock,
                         JITDylibState &JDS, MachOJITDylibDepInfoMap &DepInfo);

  Error dlcloseImpl(void *DSOHandle);
  Error dlcloseDeinitialize(std::unique_lock<std::mutex> &JDStatesLock,
                            JITDylibState &JDS);

  static MachOPlatformRuntimeState *MOPS;

  // FIXME: Move to thread-state.
  std::string DLFcnError;

  // APIMutex guards against concurrent entry into key "dyld" API functions
  // (e.g. dlopen, dlclose).
  std::recursive_mutex DyldAPIMutex;

  // JDStatesMutex guards the data structures that hold JITDylib state.
  std::mutex JDStatesMutex;
  std::unordered_map<void *, JITDylibState> JDStates;
  std::unordered_map<std::string_view, void *> JDNameToHeader;

  // ThreadDataSectionsMutex guards thread local data section state.
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
  std::lock_guard<std::mutex> Lock(JDStatesMutex);
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
  std::lock_guard<std::mutex> Lock(JDStatesMutex);
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
    std::vector<std::pair<std::string_view, ExecutorAddrRange>> Secs) {

  // FIXME: Reject platform section registration after the JITDylib is
  // sealed?

  ORC_RT_DEBUG({
    printdbg("MachOPlatform: Registering object sections for %p.\n",
             HeaderAddr.toPtr<void *>());
  });

  std::lock_guard<std::mutex> Lock(JDStatesMutex);
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
    if (KV.first == "__DATA,__data") {
      assert(!JDS->DataSectionContent.count(KV.second.Start.toPtr<char *>()) &&
             "Address already registered.");
      auto S = KV.second.toSpan<char>();
      JDS->DataSectionContent[KV.second.Start.toPtr<char *>()] =
          std::vector<char>(S.begin(), S.end());
    } else if (KV.first == "__DATA,__common") {
      // fprintf(stderr, "Adding zero-init range %llx -- %llx\n",
      // KV.second.Start.getValue(), KV.second.size());
      JDS->ZeroInitRanges[KV.second.Start.toPtr<char *>()] = KV.second.size();
    } else if (KV.first == "__DATA,__thread_data") {
      if (auto Err = registerThreadDataSection(KV.second.toSpan<const char>()))
        return Err;
    } else if (KV.first == "__DATA,__objc_selrefs")
      JDS->ObjCSelRefsSections.add(KV.second.toSpan<void *>());
    else if (KV.first == "__DATA,__objc_classlist")
      JDS->ObjCClassListSections.add(KV.second.toSpan<void *>());
    else if (KV.first == "__TEXT,__swift5_protos")
      JDS->Swift5ProtocolsSections.add(KV.second.toSpan<char>());
    else if (KV.first == "__TEXT,__swift5_proto")
      JDS->Swift5ProtocolConformancesSections.add(KV.second.toSpan<char>());
    else if (KV.first == "__TEXT,__swift5_types")
      JDS->Swift5TypesSections.add(KV.second.toSpan<char>());
    else if (KV.first == "__DATA,__mod_init_func")
      JDS->ModInitsSections.add(KV.second.toSpan<void (*)()>());
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

Error MachOPlatformRuntimeState::deregisterObjectPlatformSections(
    ExecutorAddr HeaderAddr,
    std::vector<std::pair<std::string_view, ExecutorAddrRange>> Secs) {
  // TODO: Make this more efficient? (maybe unnecessary if removal is rare?)
  // TODO: Add a JITDylib prepare-for-teardown operation that clears all
  //       registered sections, causing this function to take the fast-path.
  ORC_RT_DEBUG({
    printdbg("MachOPlatform: Registering object sections for %p.\n",
             HeaderAddr.toPtr<void *>());
  });

  std::lock_guard<std::mutex> Lock(JDStatesMutex);
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

  // TODO: Make library permanent (i.e. not able to be dlclosed) if it contains
  // any Swift or ObjC. Once this happens we can clear (and no longer record)
  // data section content, as the library could never be re-initialized.

  for (auto &KV : Secs) {
    // FIXME: Validate section ranges?
    if (KV.first == "__DATA,__data") {
      JDS->DataSectionContent.erase(KV.second.Start.toPtr<char *>());
    } else if (KV.first == "__DATA,__common") {
      JDS->ZeroInitRanges.erase(KV.second.Start.toPtr<char *>());
    } else if (KV.first == "__DATA,__thread_data") {
      if (auto Err =
              deregisterThreadDataSection(KV.second.toSpan<const char>()))
        return Err;
    } else if (KV.first == "__DATA,__objc_selrefs")
      JDS->ObjCSelRefsSections.removeIfPresent(KV.second);
    else if (KV.first == "__DATA,__objc_classlist")
      JDS->ObjCClassListSections.removeIfPresent(KV.second);
    else if (KV.first == "__TEXT,__swift5_protos")
      JDS->Swift5ProtocolsSections.removeIfPresent(KV.second);
    else if (KV.first == "__TEXT,__swift5_proto")
      JDS->Swift5ProtocolConformancesSections.removeIfPresent(KV.second);
    else if (KV.first == "__TEXT,__swift5_types")
      JDS->Swift5TypesSections.removeIfPresent(KV.second);
    else if (KV.first == "__DATA,__mod_init_func")
      JDS->ModInitsSections.removeIfPresent(KV.second);
    else {
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

void *MachOPlatformRuntimeState::dlopen(std::string_view Path, int Mode) {
  ORC_RT_DEBUG({
    std::string S(Path.data(), Path.size());
    printdbg("MachOPlatform::dlopen(\"%s\")\n", S.c_str());
  });
  std::lock_guard<std::recursive_mutex> Lock(DyldAPIMutex);
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
  std::lock_guard<std::recursive_mutex> Lock(DyldAPIMutex);
  if (auto Err = dlcloseImpl(DSOHandle)) {
    // FIXME: Make dlerror thread safe.
    DLFcnError = toString(std::move(Err));
    return -1;
  }
  return 0;
}

void *MachOPlatformRuntimeState::dlsym(void *DSOHandle,
                                       std::string_view Symbol) {
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
  std::lock_guard<std::mutex> Lock(JDStatesMutex);
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

void MachOPlatformRuntimeState::runAtExits(
    std::unique_lock<std::mutex> &JDStatesLock, JITDylibState &JDS) {
  auto AtExits = std::move(JDS.AtExits);

  // Unlock while running atexits, as they may trigger operations that modify
  // JDStates.
  JDStatesLock.unlock();
  while (!AtExits.empty()) {
    auto &AE = AtExits.back();
    AE.Func(AE.Arg);
    AtExits.pop_back();
  }
  JDStatesLock.lock();
}

void MachOPlatformRuntimeState::runAtExits(void *DSOHandle) {
  std::unique_lock<std::mutex> Lock(JDStatesMutex);
  auto *JDS = getJITDylibStateByHeader(DSOHandle);
  ORC_RT_DEBUG({
    printdbg("MachOPlatformRuntimeState::runAtExits called on unrecognized "
             "dso_handle %p\n",
             DSOHandle);
  });
  if (JDS)
    runAtExits(Lock, *JDS);
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
MachOPlatformRuntimeState::getJITDylibStateByName(std::string_view Name) {
  // FIXME: Avoid creating string once we have C++20.
  auto I = JDNameToHeader.find(std::string(Name.data(), Name.size()));
  if (I != JDNameToHeader.end())
    return getJITDylibStateByHeader(I->second);
  return nullptr;
}

Expected<ExecutorAddr>
MachOPlatformRuntimeState::lookupSymbolInJITDylib(void *DSOHandle,
                                                  std::string_view Sym) {
  Expected<ExecutorAddr> Result((ExecutorAddr()));
  if (auto Err = WrapperFunction<SPSExpected<SPSExecutorAddr>(
          SPSExecutorAddr, SPSString)>::call(&__orc_rt_macho_symbol_lookup_tag,
                                             Result,
                                             ExecutorAddr::fromPtr(DSOHandle),
                                             Sym))
    return std::move(Err);
  return Result;
}

Error MachOPlatformRuntimeState::registerObjCSelectors(JITDylibState &JDS) {
  if (!JDS.ObjCSelRefsSections.hasNewSections())
    return Error::success();

  if (ORC_RT_UNLIKELY(!sel_registerName))
    return make_error<StringError>("sel_registerName is not available");

  JDS.ObjCSelRefsSections.processNewSections([](span<void *> SelRefs) {
    for (void *&SelEntry : SelRefs) {
      const char *SelName = reinterpret_cast<const char *>(SelEntry);
      auto Sel = sel_registerName(SelName);
      *reinterpret_cast<SEL *>(&SelEntry) = Sel;
    }
  });

  return Error::success();
}

Error MachOPlatformRuntimeState::registerObjCClasses(JITDylibState &JDS) {
  if (!JDS.ObjCClassListSections.hasNewSections())
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

  return JDS.ObjCClassListSections.processNewSections(
      [&](span<void *> ClassPtrs) -> Error {
        for (void *ClassPtr : ClassPtrs) {
          auto *Cls = reinterpret_cast<Class>(ClassPtr);
          auto *ClassCompiled = reinterpret_cast<ObjCClassCompiled *>(ClassPtr);
          objc_msgSend(reinterpret_cast<id>(ClassCompiled->Parent),
                       ClassSelector);
          auto Registered = objc_readClassPair(Cls, JDS.ObjCImageInfo);
          // FIXME: Improve diagnostic by reporting the failed class's name.
          if (Registered != Cls)
            return make_error<StringError>(
                "Unable to register Objective-C class");
        }
        return Error::success();
      });
}

Error MachOPlatformRuntimeState::registerSwift5Protocols(JITDylibState &JDS) {

  if (!JDS.Swift5ProtocolsSections.hasNewSections())
    return Error::success();

  if (ORC_RT_UNLIKELY(!swift_registerProtocols))
    return make_error<StringError>("swift_registerProtocols is not available");

  JDS.Swift5ProtocolsSections.processNewSections([](span<char> ProtoSec) {
    swift_registerProtocols(
        reinterpret_cast<const ProtocolRecord *>(ProtoSec.data()),
        reinterpret_cast<const ProtocolRecord *>(ProtoSec.data() +
                                                 ProtoSec.size()));
  });

  return Error::success();
}

Error MachOPlatformRuntimeState::registerSwift5ProtocolConformances(
    JITDylibState &JDS) {

  if (!JDS.Swift5ProtocolConformancesSections.hasNewSections())
    return Error::success();

  if (ORC_RT_UNLIKELY(!swift_registerProtocolConformances))
    return make_error<StringError>(
        "swift_registerProtocolConformances is not available");

  JDS.Swift5ProtocolConformancesSections.processNewSections(
      [](span<char> ProtoConfSec) {
        swift_registerProtocolConformances(
            reinterpret_cast<const ProtocolConformanceRecord *>(
                ProtoConfSec.data()),
            reinterpret_cast<const ProtocolConformanceRecord *>(
                ProtoConfSec.data() + ProtoConfSec.size()));
      });

  return Error::success();
}

Error MachOPlatformRuntimeState::registerSwift5Types(JITDylibState &JDS) {

  if (!JDS.Swift5TypesSections.hasNewSections())
    return Error::success();

  if (ORC_RT_UNLIKELY(!swift_registerTypeMetadataRecords))
    return make_error<StringError>(
        "swift_registerTypeMetadataRecords is not available");

  JDS.Swift5TypesSections.processNewSections([&](span<char> TypesSec) {
    swift_registerTypeMetadataRecords(
        reinterpret_cast<const TypeMetadataRecord *>(TypesSec.data()),
        reinterpret_cast<const TypeMetadataRecord *>(TypesSec.data() +
                                                     TypesSec.size()));
  });

  return Error::success();
}

Error MachOPlatformRuntimeState::runModInits(
    std::unique_lock<std::mutex> &JDStatesLock, JITDylibState &JDS) {
  std::vector<span<void (*)()>> InitSections;
  InitSections.reserve(JDS.ModInitsSections.numNewSections());

  // Copy initializer sections: If the JITDylib is unsealed then the
  // initializers could reach back into the JIT and cause more initializers to
  // be added.
  // FIXME: Skip unlock and run in-place on sealed JITDylibs?
  JDS.ModInitsSections.processNewSections(
      [&](span<void (*)()> Inits) { InitSections.push_back(Inits); });

  JDStatesLock.unlock();
  for (auto InitSec : InitSections)
    for (auto *Init : InitSec)
      Init();
  JDStatesLock.lock();

  return Error::success();
}

Expected<void *> MachOPlatformRuntimeState::dlopenImpl(std::string_view Path,
                                                       int Mode) {
  std::unique_lock<std::mutex> Lock(JDStatesMutex);

  // Try to find JITDylib state by name.
  auto *JDS = getJITDylibStateByName(Path);

  if (!JDS)
    return make_error<StringError>("No registered JTIDylib for path " +
                                   std::string(Path.data(), Path.size()));

  // If this JITDylib is unsealed, or this is the first dlopen then run
  // full dlopen path (update deps, push and run initializers, update ref
  // counts on all JITDylibs in the dep tree).
  if (!JDS->referenced() || !JDS->Sealed) {
    if (auto Err = dlopenFull(Lock, *JDS))
      return std::move(Err);
  }

  // Bump the ref-count on this dylib.
  ++JDS->DlRefCount;

  // Return the header address.
  return JDS->Header;
}

Error MachOPlatformRuntimeState::dlopenFull(
    std::unique_lock<std::mutex> &JDStatesLock, JITDylibState &JDS) {
  // Call back to the JIT to push the initializers.
  Expected<MachOJITDylibDepInfoMap> DepInfo((MachOJITDylibDepInfoMap()));
  // Unlock so that we can accept the initializer update.
  JDStatesLock.unlock();
  if (auto Err = WrapperFunction<SPSExpected<SPSMachOJITDylibDepInfoMap>(
          SPSExecutorAddr)>::call(&__orc_rt_macho_push_initializers_tag,
                                  DepInfo, ExecutorAddr::fromPtr(JDS.Header)))
    return Err;
  JDStatesLock.lock();

  if (!DepInfo)
    return DepInfo.takeError();

  if (auto Err = dlopenInitialize(JDStatesLock, JDS, *DepInfo))
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
    std::unique_lock<std::mutex> &JDStatesLock, JITDylibState &JDS,
    MachOJITDylibDepInfoMap &DepInfo) {
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
    if (auto Err = dlopenInitialize(JDStatesLock, *DepJDS, DepInfo))
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
  if (auto Err = runModInits(JDStatesLock, JDS))
    return Err;

  // Decrement old deps.
  // FIXME: We should probably continue and just report deinitialize errors
  // here.
  for (auto *DepJDS : OldDeps) {
    --DepJDS->LinkedAgainstRefCount;
    if (!DepJDS->referenced())
      if (auto Err = dlcloseDeinitialize(JDStatesLock, *DepJDS))
        return Err;
  }

  return Error::success();
}

Error MachOPlatformRuntimeState::dlcloseImpl(void *DSOHandle) {
  std::unique_lock<std::mutex> Lock(JDStatesMutex);

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
    return dlcloseDeinitialize(Lock, *JDS);

  return Error::success();
}

Error MachOPlatformRuntimeState::dlcloseDeinitialize(
    std::unique_lock<std::mutex> &JDStatesLock, JITDylibState &JDS) {

  ORC_RT_DEBUG({
    printdbg("MachOPlatformRuntimeState::dlcloseDeinitialize(\"%s\")\n",
             JDS.Name.c_str());
  });

  runAtExits(JDStatesLock, JDS);

  // Reset mod-inits
  JDS.ModInitsSections.reset();

  // Reset data section contents.
  for (auto &KV : JDS.DataSectionContent)
    memcpy(KV.first, KV.second.data(), KV.second.size());
  for (auto &KV : JDS.ZeroInitRanges)
    memset(KV.first, 0, KV.second);

  // Deinitialize any dependencies.
  for (auto *DepJDS : JDS.Deps) {
    --DepJDS->LinkedAgainstRefCount;
    if (!DepJDS->referenced())
      if (auto Err = dlcloseDeinitialize(JDStatesLock, *DepJDS))
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
                std::vector<std::pair<std::string_view, ExecutorAddrRange>>
                    &Secs) {
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
                std::vector<std::pair<std::string_view, ExecutorAddrRange>>
                    &Secs) {
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
