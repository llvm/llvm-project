//===-- xray_arm.cc ---------------------------------------------*- C++ -*-===//
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
// Implementation of ARM-specific routines (32-bit).
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
  PO_PushR0Lr = 0xE92D4001, // PUSH {r0, lr}
  PO_BlxIp = 0xE12FFF3C,    // BLX ip
  PO_PopR0Lr = 0xE8BD4001,  // POP {r0, lr}
  PO_B20 = 0xEA000005       // B #20
};

// 0xUUUUWXYZ -> 0x000W0XYZ
inline static uint32_t getMovwMask(const uint32_t Value) XRAY_NEVER_INSTRUMENT {
  return (Value & 0xfff) | ((Value & 0xf000) << 4);
}

// 0xWXYZUUUU -> 0x000W0XYZ
inline static uint32_t getMovtMask(const uint32_t Value) XRAY_NEVER_INSTRUMENT {
  return getMovwMask(Value >> 16);
}

// Writes the following instructions:
//   MOVW R<regNo>, #<lower 16 bits of the |Value|>
//   MOVT R<regNo>, #<higher 16 bits of the |Value|>
inline static uint32_t *
write32bitLoadReg(uint8_t regNo, uint32_t *Address,
                  const uint32_t Value) XRAY_NEVER_INSTRUMENT {
  // This is a fatal error: we cannot just report it and continue execution.
  assert(regNo <= 15 && "Register number must be 0 to 15.");
  // MOVW R, #0xWXYZ in machine code is 0xE30WRXYZ
  *Address = (0xE3000000 | (uint32_t(regNo) << 12) | getMovwMask(Value));
  Address++;
  // MOVT R, #0xWXYZ in machine code is 0xE34WRXYZ
  *Address = (0xE3400000 | (uint32_t(regNo) << 12) | getMovtMask(Value));
  return Address + 1;
}

// Writes the following instructions:
//   MOVW r0, #<lower 16 bits of the |Value|>
//   MOVT r0, #<higher 16 bits of the |Value|>
inline static uint32_t *
Write32bitLoadR0(uint32_t *Address,
                 const uint32_t Value) XRAY_NEVER_INSTRUMENT {
  return write32bitLoadReg(0, Address, Value);
}

// Writes the following instructions:
//   MOVW ip, #<lower 16 bits of the |Value|>
//   MOVT ip, #<higher 16 bits of the |Value|>
inline static uint32_t *
Write32bitLoadIP(uint32_t *Address,
                 const uint32_t Value) XRAY_NEVER_INSTRUMENT {
  return write32bitLoadReg(12, Address, Value);
}

inline static bool patchSled(const bool Enable, const uint32_t FuncId,
                             const XRaySledEntry &Sled,
                             void (*TracingHook)()) XRAY_NEVER_INSTRUMENT {
  // When |Enable| == true,
  // We replace the following compile-time stub (sled):
  //
  // xray_sled_n:
  //   B #20
  //   6 NOPs (24 bytes)
  //
  // With the following runtime patch:
  //
  // xray_sled_n:
  //   PUSH {r0, lr}
  //   MOVW r0, #<lower 16 bits of function ID>
  //   MOVT r0, #<higher 16 bits of function ID>
  //   MOVW ip, #<lower 16 bits of address of TracingHook>
  //   MOVT ip, #<higher 16 bits of address of TracingHook>
  //   BLX ip
  //   POP {r0, lr}
  //
  // Replacement of the first 4-byte instruction should be the last and atomic
  // operation, so that the user code which reaches the sled concurrently
  // either jumps over the whole sled, or executes the whole sled when the
  // latter is ready.
  //
  // When |Enable|==false, we set back the first instruction in the sled to be
  //   B #20

  uint32_t *FirstAddress = reinterpret_cast<uint32_t *>(Sled.Address);
  uint32_t *CurAddress = FirstAddress + 1;
  if (Enable) {
    CurAddress =
        Write32bitLoadR0(CurAddress, reinterpret_cast<uint32_t>(FuncId));
    CurAddress =
        Write32bitLoadIP(CurAddress, reinterpret_cast<uint32_t>(TracingHook));
    *CurAddress = uint32_t(PatchOpcodes::PO_BlxIp);
    CurAddress++;
    *CurAddress = uint32_t(PatchOpcodes::PO_PopR0Lr);
    CurAddress++;
    std::atomic_store_explicit(
        reinterpret_cast<std::atomic<uint32_t> *>(FirstAddress),
        uint32_t(PatchOpcodes::PO_PushR0Lr), std::memory_order_release);
  } else {
    std::atomic_store_explicit(
        reinterpret_cast<std::atomic<uint32_t> *>(FirstAddress),
        uint32_t(PatchOpcodes::PO_B20), std::memory_order_release);
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
