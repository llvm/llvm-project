//===-- xray_AArch64.cc -----------------------------------------*- C++ -*-===//
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
// Implementation of AArch64-specific routines (64-bit).
//
//===----------------------------------------------------------------------===//
#include "sanitizer_common/sanitizer_common.h"
#include "xray_defs.h"
#include "xray_emulate_tsc.h"
#include "xray_interface_internal.h"
#include <atomic>
#include <cassert>


extern "C" void __clear_cache(void* start, void* end);

namespace __xray {

uint64_t cycleFrequency() XRAY_NEVER_INSTRUMENT {
  // There is no instruction like RDTSCP in user mode on ARM.  ARM's CP15 does
  //   not have a constant frequency like TSC on x86[_64]; it may go faster or
  //   slower depending on CPU's turbo or power saving modes.  Furthermore, to
  //   read from CP15 on ARM a kernel modification or a driver is needed.
  //   We can not require this from users of compiler-rt.
  // So on ARM we use clock_gettime(2) which gives the result in nanoseconds.
  //   To get the measurements per second, we scale this by the number of
  //   nanoseconds per second, pretending that the TSC frequency is 1GHz and
  //   one TSC tick is 1 nanosecond.
  return NanosecondsPerSecond;
}

// The machine codes for some instructions used in runtime patching.
enum class PatchOpcodes : uint32_t {
  PO_StpX0X30SP_m16e = 0xA9BF7BE0, // STP X0, X30, [SP, #-16]!
  PO_LdrW0_12 = 0x18000060,        // LDR W0, #12
  PO_LdrX16_12 = 0x58000070,       // LDR X16, #12
  PO_BlrX16 = 0xD63F0200,          // BLR X16
  PO_LdpX0X30SP_16 = 0xA8C17BE0,   // LDP X0, X30, [SP], #16
  PO_B32 = 0x14000008              // B #32
};

inline static bool patchSled(const bool Enable, const uint32_t FuncId,
                             const XRaySledEntry &Sled,
                             void (*TracingHook)()) XRAY_NEVER_INSTRUMENT {
  // When |Enable| == true,
  // We replace the following compile-time stub (sled):
  //
  // xray_sled_n:
  //   B #32
  //   7 NOPs (24 bytes)
  //
  // With the following runtime patch:
  //
  // xray_sled_n:
  //   STP X0, X30, [SP, #-16]! ; PUSH {r0, lr}
  //   LDR W0, #12 ; W0 := function ID
  //   LDR X16,#12 ; X16 := address of the trampoline
  //   BLR X16
  //   ;DATA: 32 bits of function ID
  //   ;DATA: lower 32 bits of the address of the trampoline
  //   ;DATA: higher 32 bits of the address of the trampoline
  //   LDP X0, X30, [SP], #16 ; POP {r0, lr}
  //
  // Replacement of the first 4-byte instruction should be the last and atomic
  // operation, so that the user code which reaches the sled concurrently
  // either jumps over the whole sled, or executes the whole sled when the
  // latter is ready.
  //
  // When |Enable|==false, we set back the first instruction in the sled to be
  //   B #32

  uint32_t *FirstAddress = reinterpret_cast<uint32_t *>(Sled.Address);
  uint32_t *CurAddress = FirstAddress + 1;
  if (Enable) {
    *CurAddress = uint32_t(PatchOpcodes::PO_LdrW0_12);
    CurAddress++;
    *CurAddress = uint32_t(PatchOpcodes::PO_LdrX16_12);
    CurAddress++;
    *CurAddress = uint32_t(PatchOpcodes::PO_BlrX16);
    CurAddress++;
    *CurAddress = FuncId;
    CurAddress++;
    *reinterpret_cast<void (**)()>(CurAddress) = TracingHook;
    CurAddress += 2;
    *CurAddress = uint32_t(PatchOpcodes::PO_LdpX0X30SP_16);
    CurAddress++;
    std::atomic_store_explicit(
        reinterpret_cast<std::atomic<uint32_t> *>(FirstAddress),
        uint32_t(PatchOpcodes::PO_StpX0X30SP_m16e), std::memory_order_release);
  } else {
    std::atomic_store_explicit(
        reinterpret_cast<std::atomic<uint32_t> *>(FirstAddress),
        uint32_t(PatchOpcodes::PO_B32), std::memory_order_release);
  }
  __clear_cache(reinterpret_cast<char*>(FirstAddress),
      reinterpret_cast<char*>(CurAddress));
  return true;
}

bool patchFunctionEntry(const bool Enable, const uint32_t FuncId,
                        const XRaySledEntry &Sled) XRAY_NEVER_INSTRUMENT {
  return patchSled(Enable, FuncId, Sled, __xray_FunctionEntry);
}

bool patchFunctionExit(const bool Enable, const uint32_t FuncId,
                       const XRaySledEntry &Sled) XRAY_NEVER_INSTRUMENT {
  return patchSled(Enable, FuncId, Sled, __xray_FunctionExit);
}

bool patchFunctionTailExit(const bool Enable, const uint32_t FuncId,
                           const XRaySledEntry &Sled) XRAY_NEVER_INSTRUMENT {
  // FIXME: In the future we'd need to distinguish between non-tail exits and
  // tail exits for better information preservation.
  return patchSled(Enable, FuncId, Sled, __xray_FunctionExit);
}

} // namespace __xray
