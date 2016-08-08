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

namespace __xray {

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
  explicit MProtectHelper(void *PageAlignedAddr, std::size_t MProtectLen)
      : PageAlignedAddr(PageAlignedAddr), MProtectLen(MProtectLen),
        MustCleanup(false) {}

  int MakeWriteable() {
    auto R = mprotect(PageAlignedAddr, MProtectLen,
                      PROT_READ | PROT_WRITE | PROT_EXEC);
    if (R != -1)
      MustCleanup = true;
    return R;
  }

  ~MProtectHelper() {
    if (MustCleanup) {
      mprotect(PageAlignedAddr, MProtectLen, PROT_READ | PROT_EXEC);
    }
  }
};

} // namespace __xray

extern "C" {
// The following functions have to be defined in assembler, on a per-platform
// basis. See xray_trampoline_*.s files for implementations.
extern void __xray_FunctionEntry();
extern void __xray_FunctionExit();
}

extern std::atomic<bool> XRayInitialized;
extern std::atomic<__xray::XRaySledMap> XRayInstrMap;

int __xray_set_handler(void (*entry)(int32_t, XRayEntryType)) {
  if (XRayInitialized.load(std::memory_order_acquire)) {
    __xray::XRayPatchedFunction.store(entry, std::memory_order_release);
    return 1;
  }
  return 0;
}

int __xray_remove_handler() { return __xray_set_handler(nullptr); }

std::atomic<bool> XRayPatching{false};

using namespace __xray;

// FIXME: Figure out whether we can move this class to sanitizer_common instead
// as a generic "scope guard".
template <class Function> class CleanupInvoker {
  Function Fn;

public:
  explicit CleanupInvoker(Function Fn) : Fn(Fn) {}
  CleanupInvoker(const CleanupInvoker &) = default;
  CleanupInvoker(CleanupInvoker &&) = default;
  CleanupInvoker &operator=(const CleanupInvoker &) = delete;
  CleanupInvoker &operator=(CleanupInvoker &&) = delete;
  ~CleanupInvoker() { Fn(); }
};

template <class Function> CleanupInvoker<Function> ScopeCleanup(Function Fn) {
  return CleanupInvoker<Function>{Fn};
}

// ControlPatching implements the common internals of the patching/unpatching
// implementation. |Enable| defines whether we're enabling or disabling the
// runtime XRay instrumentation.
XRayPatchingStatus ControlPatching(bool Enable) {
  if (!XRayInitialized.load(std::memory_order_acquire))
    return XRayPatchingStatus::NOT_INITIALIZED; // Not initialized.

  static bool NotPatching = false;
  if (!XRayPatching.compare_exchange_strong(NotPatching, true,
                                            std::memory_order_acq_rel,
                                            std::memory_order_acquire)) {
    return XRayPatchingStatus::ONGOING; // Already patching.
  }

  bool PatchingSuccess = false;
  auto XRayPatchingStatusResetter = ScopeCleanup([&PatchingSuccess] {
    if (!PatchingSuccess) {
      XRayPatching.store(false, std::memory_order_release);
    }
  });

  // Step 1: Compute the function id, as a unique identifier per function in the
  // instrumentation map.
  XRaySledMap InstrMap = XRayInstrMap.load(std::memory_order_acquire);
  if (InstrMap.Entries == 0)
    return XRayPatchingStatus::NOT_INITIALIZED;

  int32_t FuncId = 1;
  static constexpr uint8_t CallOpCode = 0xe8;
  static constexpr uint16_t MovR10Seq = 0xba41;
  static constexpr uint16_t Jmp9Seq = 0x09eb;
  static constexpr uint8_t JmpOpCode = 0xe9;
  static constexpr uint8_t RetOpCode = 0xc3;
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
        reinterpret_cast<void *>(Sled.Address & ~((2 << 16) - 1));
    std::size_t MProtectLen =
        (Sled.Address + 12) - reinterpret_cast<uint64_t>(PageAlignedAddr);
    MProtectHelper Protector(PageAlignedAddr, MProtectLen);
    if (Protector.MakeWriteable() == -1) {
      printf("Failed mprotect: %d\n", errno);
      return XRayPatchingStatus::FAILED;
    }

    static constexpr int64_t MinOffset{std::numeric_limits<int32_t>::min()};
    static constexpr int64_t MaxOffset{std::numeric_limits<int32_t>::max()};
    if (Sled.Kind == XRayEntryType::ENTRY) {
      // FIXME: Implement this in a more extensible manner, per-platform.
      // Here we do the dance of replacing the following sled:
      //
      // xray_sled_n:
      //   jmp +9
      //   <9 byte nop>
      //
      // With the following:
      //
      //   mov r10d, <function id>
      //   call <relative 32bit offset to entry trampoline>
      //
      // We need to do this in the following order:
      //
      // 1. Put the function id first, 2 bytes from the start of the sled (just
      // after the 2-byte jmp instruction).
      // 2. Put the call opcode 6 bytes from the start of the sled.
      // 3. Put the relative offset 7 bytes from the start of the sled.
      // 4. Do an atomic write over the jmp instruction for the "mov r10d"
      // opcode and first operand.
      //
      // Prerequisite is to compute the relative offset to the
      // __xray_FunctionEntry function's address.
      int64_t TrampolineOffset =
          reinterpret_cast<int64_t>(__xray_FunctionEntry) -
          (static_cast<int64_t>(Sled.Address) + 11);
      if (TrampolineOffset < MinOffset || TrampolineOffset > MaxOffset) {
        Report("XRay Entry trampoline (%p) too far from sled (%p); distance = "
               "%ld\n",
               __xray_FunctionEntry, reinterpret_cast<void *>(Sled.Address),
               TrampolineOffset);
        continue;
      }
      if (Enable) {
        *reinterpret_cast<uint32_t *>(Sled.Address + 2) = FuncId;
        *reinterpret_cast<uint8_t *>(Sled.Address + 6) = CallOpCode;
        *reinterpret_cast<uint32_t *>(Sled.Address + 7) = TrampolineOffset;
        std::atomic_store_explicit(
            reinterpret_cast<std::atomic<uint16_t> *>(Sled.Address), MovR10Seq,
            std::memory_order_release);
      } else {
        std::atomic_store_explicit(
            reinterpret_cast<std::atomic<uint16_t> *>(Sled.Address), Jmp9Seq,
            std::memory_order_release);
        // FIXME: Write out the nops still?
      }
    }

    if (Sled.Kind == XRayEntryType::EXIT) {
      // FIXME: Implement this in a more extensible manner, per-platform.
      // Here we do the dance of replacing the following sled:
      //
      // xray_sled_n:
      //   ret
      //   <10 byte nop>
      //
      // With the following:
      //
      //   mov r10d, <function id>
      //   jmp <relative 32bit offset to exit trampoline>
      //
      // 1. Put the function id first, 2 bytes from the start of the sled (just
      // after the 1-byte ret instruction).
      // 2. Put the jmp opcode 6 bytes from the start of the sled.
      // 3. Put the relative offset 7 bytes from the start of the sled.
      // 4. Do an atomic write over the jmp instruction for the "mov r10d"
      // opcode and first operand.
      //
      // Prerequisite is to compute the relative offset fo the
      // __xray_FunctionExit function's address.
      int64_t TrampolineOffset =
          reinterpret_cast<int64_t>(__xray_FunctionExit) -
          (static_cast<int64_t>(Sled.Address) + 11);
      if (TrampolineOffset < MinOffset || TrampolineOffset > MaxOffset) {
        Report("XRay Exit trampoline (%p) too far from sled (%p); distance = "
               "%ld\n",
               __xray_FunctionExit, reinterpret_cast<void *>(Sled.Address),
               TrampolineOffset);
        continue;
      }
      if (Enable) {
        *reinterpret_cast<uint32_t *>(Sled.Address + 2) = FuncId;
        *reinterpret_cast<uint8_t *>(Sled.Address + 6) = JmpOpCode;
        *reinterpret_cast<uint32_t *>(Sled.Address + 7) = TrampolineOffset;
        std::atomic_store_explicit(
            reinterpret_cast<std::atomic<uint16_t> *>(Sled.Address), MovR10Seq,
            std::memory_order_release);
      } else {
        std::atomic_store_explicit(
            reinterpret_cast<std::atomic<uint8_t> *>(Sled.Address), RetOpCode,
            std::memory_order_release);
        // FIXME: Write out the nops still?
      }
    }
  }
  XRayPatching.store(false, std::memory_order_release);
  PatchingSuccess = true;
  return XRayPatchingStatus::SUCCESS;
}

XRayPatchingStatus __xray_patch() { return ControlPatching(true); }

XRayPatchingStatus __xray_unpatch() { return ControlPatching(false); }
