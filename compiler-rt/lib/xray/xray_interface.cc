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

#include <atomic>
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
#else
#error "Unsupported CPU Architecture"
#endif /* CPU architecture */

// This is the function to call when we encounter the entry or exit sleds.
std::atomic<void (*)(int32_t, XRayEntryType)> XRayPatchedFunction{nullptr};

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

extern std::atomic<bool> XRayInitialized;
extern std::atomic<__xray::XRaySledMap> XRayInstrMap;

int __xray_set_handler(void (*entry)(int32_t,
                                     XRayEntryType)) XRAY_NEVER_INSTRUMENT {
  if (XRayInitialized.load(std::memory_order_acquire)) {
    __xray::XRayPatchedFunction.store(entry, std::memory_order_release);
    return 1;
  }
  return 0;
}

int __xray_remove_handler() XRAY_NEVER_INSTRUMENT {
  return __xray_set_handler(nullptr);
}

std::atomic<bool> XRayPatching{false};

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

// controlPatching implements the common internals of the patching/unpatching
// implementation. |Enable| defines whether we're enabling or disabling the
// runtime XRay instrumentation.
XRayPatchingStatus controlPatching(bool Enable) XRAY_NEVER_INSTRUMENT {
  if (!XRayInitialized.load(std::memory_order_acquire))
    return XRayPatchingStatus::NOT_INITIALIZED; // Not initialized.

  static bool NotPatching = false;
  if (!XRayPatching.compare_exchange_strong(NotPatching, true,
                                            std::memory_order_acq_rel,
                                            std::memory_order_acquire)) {
    return XRayPatchingStatus::ONGOING; // Already patching.
  }

  bool PatchingSuccess = false;
  auto XRayPatchingStatusResetter = scopeCleanup([&PatchingSuccess] {
    if (!PatchingSuccess) {
      XRayPatching.store(false, std::memory_order_release);
    }
  });

  // Step 1: Compute the function id, as a unique identifier per function in the
  // instrumentation map.
  XRaySledMap InstrMap = XRayInstrMap.load(std::memory_order_acquire);
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

    // While we're here, we should patch the nop sled. To do that we mprotect
    // the page containing the function to be writeable.
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
      Success = patchFunctionEntry(Enable, FuncId, Sled);
      break;
    case XRayEntryType::EXIT:
      Success = patchFunctionExit(Enable, FuncId, Sled);
      break;
    case XRayEntryType::TAIL:
      Success = patchFunctionTailExit(Enable, FuncId, Sled);
      break;
    default:
      Report("Unsupported sled kind: %d\n", int(Sled.Kind));
      continue;
    }
    (void)Success;
  }
  XRayPatching.store(false, std::memory_order_release);
  PatchingSuccess = true;
  return XRayPatchingStatus::SUCCESS;
}

XRayPatchingStatus __xray_patch() XRAY_NEVER_INSTRUMENT {
  return controlPatching(true);
}

XRayPatchingStatus __xray_unpatch() XRAY_NEVER_INSTRUMENT {
  return controlPatching(false);
}
