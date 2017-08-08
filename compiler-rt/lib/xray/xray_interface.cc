//===-- xray_interface.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of XRay, a dynamic runtime instrumentation system.
//
// Implementation of the API functions.
//
//===----------------------------------------------------------------------===//

#include "xray_interface_internal.h"

#include <cstdint>
#include <cstdio>
#include <errno.h>
#include <limits>
#include <sys/mman.h>

#include "sanitizer_common/sanitizer_common.h"
#include "xray_defs.h"

namespace __xray {

#if defined(__x86_64__)
// FIXME: The actual length is 11 bytes. Why was length 12 passed to mprotect()
// ?
static const int16_t cSledLength = 12;
#elif defined(__aarch64__)
static const int16_t cSledLength = 32;
#elif defined(__arm__)
static const int16_t cSledLength = 28;
#elif SANITIZER_MIPS32
static const int16_t cSledLength = 48;
#elif SANITIZER_MIPS64
static const int16_t cSledLength = 64;
#elif defined(__powerpc64__)
static const int16_t cSledLength = 8;
#else
#error "Unsupported CPU Architecture"
#endif /* CPU architecture */

// This is the function to call when we encounter the entry or exit sleds.
__sanitizer::atomic_uintptr_t XRayPatchedFunction{0};

// This is the function to call from the arg1-enabled sleds/trampolines.
__sanitizer::atomic_uintptr_t XRayArgLogger{0};

// This is the function to call when we encounter a custom event log call.
__sanitizer::atomic_uintptr_t XRayPatchedCustomEvent{0};

// MProtectHelper is an RAII wrapper for calls to mprotect(...) that will undo
// any successful mprotect(...) changes. This is used to make a page writeable
// and executable, and upon destruction if it was successful in doing so returns
// the page into a read-only and executable page.
//
// This is only used specifically for runtime-patching of the XRay
// instrumentation points. This assumes that the executable pages are originally
// read-and-execute only.
class MProtectHelper {
  void *PageAlignedAddr;
  std::size_t MProtectLen;
  bool MustCleanup;

public:
  explicit MProtectHelper(void *PageAlignedAddr,
                          std::size_t MProtectLen) XRAY_NEVER_INSTRUMENT
      : PageAlignedAddr(PageAlignedAddr),
        MProtectLen(MProtectLen),
        MustCleanup(false) {}

  int MakeWriteable() XRAY_NEVER_INSTRUMENT {
    auto R = mprotect(PageAlignedAddr, MProtectLen,
                      PROT_READ | PROT_WRITE | PROT_EXEC);
    if (R != -1)
      MustCleanup = true;
    return R;
  }

  ~MProtectHelper() XRAY_NEVER_INSTRUMENT {
    if (MustCleanup) {
      mprotect(PageAlignedAddr, MProtectLen, PROT_READ | PROT_EXEC);
    }
  }
};

} // namespace __xray

extern __sanitizer::SpinMutex XRayInstrMapMutex;
extern __sanitizer::atomic_uint8_t XRayInitialized;
extern __xray::XRaySledMap XRayInstrMap;

int __xray_set_handler(void (*entry)(int32_t,
                                     XRayEntryType)) XRAY_NEVER_INSTRUMENT {
  if (__sanitizer::atomic_load(&XRayInitialized,
                               __sanitizer::memory_order_acquire)) {

    __sanitizer::atomic_store(&__xray::XRayPatchedFunction,
                              reinterpret_cast<uintptr_t>(entry),
                              __sanitizer::memory_order_release);
    return 1;
  }
  return 0;
}

int __xray_set_customevent_handler(void (*entry)(void *, size_t))
    XRAY_NEVER_INSTRUMENT {
  if (__sanitizer::atomic_load(&XRayInitialized,
                               __sanitizer::memory_order_acquire)) {
    __sanitizer::atomic_store(&__xray::XRayPatchedCustomEvent,
                              reinterpret_cast<uintptr_t>(entry),
                              __sanitizer::memory_order_release);
    return 1;
  }
  return 0;
}


int __xray_remove_handler() XRAY_NEVER_INSTRUMENT {
  return __xray_set_handler(nullptr);
}

int __xray_remove_customevent_handler() XRAY_NEVER_INSTRUMENT {
  return __xray_set_customevent_handler(nullptr);
}

__sanitizer::atomic_uint8_t XRayPatching{0};

using namespace __xray;

// FIXME: Figure out whether we can move this class to sanitizer_common instead
// as a generic "scope guard".
template <class Function> class CleanupInvoker {
  Function Fn;

public:
  explicit CleanupInvoker(Function Fn) XRAY_NEVER_INSTRUMENT : Fn(Fn) {}
  CleanupInvoker(const CleanupInvoker &) XRAY_NEVER_INSTRUMENT = default;
  CleanupInvoker(CleanupInvoker &&) XRAY_NEVER_INSTRUMENT = default;
  CleanupInvoker &
  operator=(const CleanupInvoker &) XRAY_NEVER_INSTRUMENT = delete;
  CleanupInvoker &operator=(CleanupInvoker &&) XRAY_NEVER_INSTRUMENT = delete;
  ~CleanupInvoker() XRAY_NEVER_INSTRUMENT { Fn(); }
};

template <class Function>
CleanupInvoker<Function> scopeCleanup(Function Fn) XRAY_NEVER_INSTRUMENT {
  return CleanupInvoker<Function>{Fn};
}

inline bool patchSled(const XRaySledEntry &Sled, bool Enable,
                      int32_t FuncId) XRAY_NEVER_INSTRUMENT {
  // While we're here, we should patch the nop sled. To do that we mprotect
  // the page containing the function to be writeable.
  const uint64_t PageSize = GetPageSizeCached();
  void *PageAlignedAddr =
      reinterpret_cast<void *>(Sled.Address & ~(PageSize - 1));
  std::size_t MProtectLen = (Sled.Address + cSledLength) -
                            reinterpret_cast<uint64_t>(PageAlignedAddr);
  MProtectHelper Protector(PageAlignedAddr, MProtectLen);
  if (Protector.MakeWriteable() == -1) {
    printf("Failed mprotect: %d\n", errno);
    return XRayPatchingStatus::FAILED;
  }

  bool Success = false;
  switch (Sled.Kind) {
  case XRayEntryType::ENTRY:
    Success = patchFunctionEntry(Enable, FuncId, Sled, __xray_FunctionEntry);
    break;
  case XRayEntryType::EXIT:
    Success = patchFunctionExit(Enable, FuncId, Sled);
    break;
  case XRayEntryType::TAIL:
    Success = patchFunctionTailExit(Enable, FuncId, Sled);
    break;
  case XRayEntryType::LOG_ARGS_ENTRY:
    Success = patchFunctionEntry(Enable, FuncId, Sled, __xray_ArgLoggerEntry);
    break;
  case XRayEntryType::CUSTOM_EVENT:
    Success = patchCustomEvent(Enable, FuncId, Sled);
    break;
  default:
    Report("Unsupported sled kind '%d' @%04x\n", Sled.Address, int(Sled.Kind));
    return false;
  }
  return Success;
}

// controlPatching implements the common internals of the patching/unpatching
// implementation. |Enable| defines whether we're enabling or disabling the
// runtime XRay instrumentation.
XRayPatchingStatus controlPatching(bool Enable) XRAY_NEVER_INSTRUMENT {
  if (!__sanitizer::atomic_load(&XRayInitialized,
                                __sanitizer::memory_order_acquire))
    return XRayPatchingStatus::NOT_INITIALIZED; // Not initialized.

  uint8_t NotPatching = false;
  if (!__sanitizer::atomic_compare_exchange_strong(
          &XRayPatching, &NotPatching, true, __sanitizer::memory_order_acq_rel))
    return XRayPatchingStatus::ONGOING; // Already patching.

  uint8_t PatchingSuccess = false;
  auto XRayPatchingStatusResetter = scopeCleanup([&PatchingSuccess] {
    if (!PatchingSuccess)
      __sanitizer::atomic_store(&XRayPatching, false,
                                __sanitizer::memory_order_release);
  });

  // Step 1: Compute the function id, as a unique identifier per function in the
  // instrumentation map.
  XRaySledMap InstrMap;
  {
    __sanitizer::SpinMutexLock Guard(&XRayInstrMapMutex);
    InstrMap = XRayInstrMap;
  }
  if (InstrMap.Entries == 0)
    return XRayPatchingStatus::NOT_INITIALIZED;

  const uint64_t PageSize = GetPageSizeCached();
  if ((PageSize == 0) || ((PageSize & (PageSize - 1)) != 0)) {
    Report("System page size is not a power of two: %lld\n", PageSize);
    return XRayPatchingStatus::FAILED;
  }

  uint32_t FuncId = 1;
  uint64_t CurFun = 0;
  for (std::size_t I = 0; I < InstrMap.Entries; I++) {
    auto Sled = InstrMap.Sleds[I];
    auto F = Sled.Function;
    if (CurFun == 0)
      CurFun = F;
    if (F != CurFun) {
      ++FuncId;
      CurFun = F;
    }
    patchSled(Sled, Enable, FuncId);
  }
  __sanitizer::atomic_store(&XRayPatching, false,
                            __sanitizer::memory_order_release);
  PatchingSuccess = true;
  return XRayPatchingStatus::SUCCESS;
}

XRayPatchingStatus __xray_patch() XRAY_NEVER_INSTRUMENT {
  return controlPatching(true);
}

XRayPatchingStatus __xray_unpatch() XRAY_NEVER_INSTRUMENT {
  return controlPatching(false);
}

XRayPatchingStatus patchFunction(int32_t FuncId,
                                 bool Enable) XRAY_NEVER_INSTRUMENT {
  if (!__sanitizer::atomic_load(&XRayInitialized,
                                __sanitizer::memory_order_acquire))
    return XRayPatchingStatus::NOT_INITIALIZED; // Not initialized.

  uint8_t NotPatching = false;
  if (!__sanitizer::atomic_compare_exchange_strong(
          &XRayPatching, &NotPatching, true, __sanitizer::memory_order_acq_rel))
    return XRayPatchingStatus::ONGOING; // Already patching.

  // Next, we look for the function index.
  XRaySledMap InstrMap;
  {
    __sanitizer::SpinMutexLock Guard(&XRayInstrMapMutex);
    InstrMap = XRayInstrMap;
  }

  // If we don't have an index, we can't patch individual functions.
  if (InstrMap.Functions == 0)
    return XRayPatchingStatus::NOT_INITIALIZED;

  // FuncId must be a positive number, less than the number of functions
  // instrumented.
  if (FuncId <= 0 || static_cast<size_t>(FuncId) > InstrMap.Functions) {
    Report("Invalid function id provided: %d\n", FuncId);
    return XRayPatchingStatus::FAILED;
  }

  // Now we patch ths sleds for this specific function.
  auto SledRange = InstrMap.SledsIndex[FuncId - 1];
  auto *f = SledRange.Begin;
  auto *e = SledRange.End;

  bool SucceedOnce = false;
  while (f != e)
    SucceedOnce |= patchSled(*f++, Enable, FuncId);

  __sanitizer::atomic_store(&XRayPatching, false,
                            __sanitizer::memory_order_release);

  if (!SucceedOnce) {
    Report("Failed patching any sled for function '%d'.", FuncId);
    return XRayPatchingStatus::FAILED;
  }

  return XRayPatchingStatus::SUCCESS;
}

XRayPatchingStatus __xray_patch_function(int32_t FuncId) XRAY_NEVER_INSTRUMENT {
  return patchFunction(FuncId, true);
}

XRayPatchingStatus
__xray_unpatch_function(int32_t FuncId) XRAY_NEVER_INSTRUMENT {
  return patchFunction(FuncId, false);
}

int __xray_set_handler_arg1(void (*entry)(int32_t, XRayEntryType, uint64_t)) {
  if (!__sanitizer::atomic_load(&XRayInitialized,
                                __sanitizer::memory_order_acquire))
    return 0;

  // A relaxed write might not be visible even if the current thread gets
  // scheduled on a different CPU/NUMA node.  We need to wait for everyone to
  // have this handler installed for consistency of collected data across CPUs.
  __sanitizer::atomic_store(&XRayArgLogger, reinterpret_cast<uint64_t>(entry),
                            __sanitizer::memory_order_release);
  return 1;
}

int __xray_remove_handler_arg1() { return __xray_set_handler_arg1(nullptr); }

uintptr_t __xray_function_address(int32_t FuncId) XRAY_NEVER_INSTRUMENT {
  __sanitizer::SpinMutexLock Guard(&XRayInstrMapMutex);
  if (FuncId <= 0 || static_cast<size_t>(FuncId) > XRayInstrMap.Functions)
    return 0;
  return XRayInstrMap.SledsIndex[FuncId - 1].Begin->Address
// On PPC, function entries are always aligned to 16 bytes. The beginning of a
// sled might be a local entry, which is always +8 based on the global entry.
// Always return the global entry.
#ifdef __PPC__
         & ~0xf
#endif
      ;
}

size_t __xray_max_function_id() XRAY_NEVER_INSTRUMENT {
  __sanitizer::SpinMutexLock Guard(&XRayInstrMapMutex);
  return XRayInstrMap.Functions;
}
