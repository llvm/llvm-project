//===-- xray_hexagon.cpp --------------------------------------*- C++ ---*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of XRay, a dynamic runtime instrumentation system.
//
// Implementation of hexagon-specific routines (32-bit).
//
//===----------------------------------------------------------------------===//
#include "sanitizer_common/sanitizer_common.h"
#include "xray_defs.h"
#include "xray_interface_internal.h"
#include <assert.h>
#include <atomic>

namespace __xray {

// The machine codes for some instructions used in runtime patching.
//
// J2_jump encoding: bits [27:16] and [13:1] hold the PC-relative byte offset
// divided by 4, with bits [15:14] = 0b11 for packet-end parse bits.
// Formula: 0x5800c000 | ((ByteOffset >> 2) << 1)
enum PatchOpcodes : uint32_t {
  PO_JUMPI_1C = 0x5800c00e,     // jump #0x01c (entry/exit sled, 28 bytes)
  PO_JUMPI_30 = 0x5800c018,     // jump #0x030 (custom event sled, 48 bytes)
  PO_JUMPI_3C = 0x5800c01e,     // jump #0x03c (typed event sled, 60 bytes)
  PO_NOP = 0x7f00c000,          // { nop } with packet-end parse bits
  PO_CALLR_R6 = 0x50a6c000,     // indirect call: callr r6
  PO_TFR_IMM = 0x78000000,      // transfer immed
                                // ICLASS 0x7 - S2-type A-type
  PO_IMMEXT = 0x00000000,       // constant extender
  PO_ALLOCFRAME_0 = 0xa09dc000, // allocframe(#0)
  PO_ALLOCFRAME_8 = 0xa09dc001, // allocframe(#8)
  PO_DEALLOCFRAME = 0x901ec01e, // deallocframe
  PO_STORE_R7_SP0 = 0xa19dc700, // memw(r29+#0) = r7
  PO_LOAD_R7_SP0 = 0x919dc007,  // r7 = memw(r29+#0)
  PO_CALL = 0x5a00c000, // call #0 (J2_call, packet end); offset added in
};

// Encode a J2_call to a PC-relative byte offset.  The offset (divided by 4) is
// a 22-bit signed value split across instruction bits [24:16] and [13:1].
inline static uint32_t encodeCall(int32_t ByteOffset) XRAY_NEVER_INSTRUMENT {
  const int32_t Imm = ByteOffset >> 2;
  return PO_CALL | ((Imm & 0x1fff) << 1) | (((Imm >> 13) & 0x1ff) << 16);
}

enum PacketWordParseBits : uint32_t {
  PP_DUPLEX = 0x00 << 14,
  PP_NOT_END = 0x01 << 14,
  PP_PACKET_END = 0x03 << 14,
};

enum RegNum : uint32_t {
  RN_R6 = 0x6,
  RN_R7 = 0x7,
};

inline static uint32_t
encodeExtendedTransferImmediate(uint32_t Imm, RegNum DestReg,
                                bool PacketEnd = false) XRAY_NEVER_INSTRUMENT {
  static const uint32_t REG_MASK = 0x1f;
  assert((DestReg & (~REG_MASK)) == 0);
  // The constant-extended register transfer encodes the 6 least
  // significant bits of the effective constant:
  Imm = Imm & 0x03f;
  const PacketWordParseBits ParseBits = PacketEnd ? PP_PACKET_END : PP_NOT_END;

  return PO_TFR_IMM | ParseBits | (Imm << 5) | (DestReg & REG_MASK);
}

inline static uint32_t
encodeConstantExtender(uint32_t Imm) XRAY_NEVER_INSTRUMENT {
  // Bits   Name      Description
  // -----  -------   ------------------------------------------
  // 31:28  ICLASS    Instruction class = 0000
  // 27:16  high      High 12 bits of 26-bit constant extension
  // 15:14  Parse     Parse bits
  // 13:0   low       Low 14 bits of 26-bit constant extension
  static const uint32_t IMM_MASK_LOW = 0x03fff;
  static const uint32_t IMM_MASK_HIGH = 0x00fff << 14;

  // The extender encodes the 26 most significant bits of the effective
  // constant:
  Imm = Imm >> 6;

  const uint32_t high = (Imm & IMM_MASK_HIGH) << 16;
  const uint32_t low = Imm & IMM_MASK_LOW;

  return PO_IMMEXT | high | PP_NOT_END | low;
}

static void WriteInstFlushCache(void *Addr, uint32_t NewInstruction) {
  asm volatile("icinva(%[inst_addr])\n\t"
               "isync\n\t"
               "memw(%[inst_addr]) = %[new_inst]\n\t"
               "dccleaninva(%[inst_addr])\n\t"
               "syncht\n\t"
               :
               : [ inst_addr ] "r"(Addr), [ new_inst ] "r"(NewInstruction)
               : "memory");
}

inline static bool patchSled(const bool Enable, const uint32_t FuncId,
                             const XRaySledEntry &Sled,
                             void (*TracingHook)()) XRAY_NEVER_INSTRUMENT {
  // When |Enable| == true,
  // We replace the following compile-time stub (sled):
  //
  // .L_xray_sled_N:
  // <xray_sled_base>:
  // { jump .Ltmp0 }
  // { nop } x 6
  // .Ltmp0:
  //
  // With the following runtime patch:
  //
  // <xray_sled_n>:
  // { allocframe(#8) }       // save r31:30, reserve an 8-byte spill slot
  // { memw(sp+#0) = r7 }     // preserve r7 (the only reg this sled clobbers)
  // { immext(#...)           // upper 26-bits of func id
  //   r7 = ##... }           // lower  6-bits of func id
  // { call ##trampoline }    // direct PC-relative call (does NOT clobber r6)
  // { r7 = memw(sp+#0) }     // restore r7
  // { deallocframe }
  //
  // An XRay sled is inserted post-register-allocation and is treated by the
  // compiler as clobbering nothing, yet a function-exit sled can land in the
  // middle of a loop (a conditional early return) where r6/r7 hold live
  // loop-carried values.  Earlier code loaded the trampoline address into r6
  // and used `callr r6`, clobbering both r6 and r7 with no way to recover the
  // originals (the clobber precedes the trampoline's register save).  Use a
  // direct `call` so r6 is never touched (the trampoline preserves the rest of
  // the caller-saved set, including r6), and have the sled itself spill/reload
  // r7 around the funcid setup.  allocframe(#8)/deallocframe save the caller's
  // r31:30 (LR:FP) and provide the spill slot.
  //
  // Replacement of the first 4-byte instruction should be the last and
  // atomic operation, so that user code reaching the sled concurrently
  // either jumps over the whole sled, or executes the whole sled when it
  // is ready.
  //
  // When |Enable|==false, we set back the first instruction in the sled to be
  // { jump .Ltmp0 }

  uint32_t *FirstAddress = reinterpret_cast<uint32_t *>(Sled.address());
  if (Enable) {
    uint32_t *CurAddress = FirstAddress + 1;
    // Word 1: memw(sp+#0) = r7  -- spill r7 before clobbering it below.
    *CurAddress = uint32_t(PO_STORE_R7_SP0);
    CurAddress++;
    // Word 2: immext for r7 = FuncId
    *CurAddress = encodeConstantExtender(FuncId);
    CurAddress++;
    // Word 3: r7 = ##FuncId (low 6 bits), packet end
    *CurAddress = encodeExtendedTransferImmediate(FuncId, RN_R7, true);
    CurAddress++;
    // Word 4: call ##TracingHook -- PC-relative from this instruction.
    *CurAddress = encodeCall(
        static_cast<int32_t>(reinterpret_cast<intptr_t>(TracingHook) -
                             reinterpret_cast<intptr_t>(CurAddress)));
    CurAddress++;
    // Word 5: r7 = memw(sp+#0)  -- restore r7.
    *CurAddress = uint32_t(PO_LOAD_R7_SP0);
    CurAddress++;
    // Word 6: deallocframe
    *CurAddress = uint32_t(PO_DEALLOCFRAME);

    // Word 0 (written last, atomically): allocframe(#8) replaces jump
    WriteInstFlushCache(FirstAddress, uint32_t(PO_ALLOCFRAME_8));
  } else {
    WriteInstFlushCache(FirstAddress, uint32_t(PO_JUMPI_1C));
  }
  return true;
}

bool patchFunctionEntry(const bool Enable, const uint32_t FuncId,
                        const XRaySledEntry &Sled,
                        const XRayTrampolines &Trampolines,
                        bool LogArgs) XRAY_NEVER_INSTRUMENT {
  auto Trampoline =
      LogArgs ? Trampolines.LogArgsTrampoline : Trampolines.EntryTrampoline;
  return patchSled(Enable, FuncId, Sled, Trampoline);
}

bool patchFunctionExit(
    const bool Enable, const uint32_t FuncId, const XRaySledEntry &Sled,
    const XRayTrampolines &Trampolines) XRAY_NEVER_INSTRUMENT {
  return patchSled(Enable, FuncId, Sled, Trampolines.ExitTrampoline);
}

bool patchFunctionTailExit(
    const bool Enable, const uint32_t FuncId, const XRaySledEntry &Sled,
    const XRayTrampolines &Trampolines) XRAY_NEVER_INSTRUMENT {
  return patchSled(Enable, FuncId, Sled, Trampolines.TailExitTrampoline);
}

bool patchCustomEvent(const bool Enable, const uint32_t FuncId,
                      const XRaySledEntry &Sled) XRAY_NEVER_INSTRUMENT {
  // The custom event sled (2 args) is 12 words = 48 bytes:
  //   .Lxray_sled_N:
  //     { jump .Lend }     <-- first word: jump over (disabled) / nop (enabled)
  //     allocframe, sp adjust, 2 saves, 2 moves, call, 2 restores,
  //     sp adjust, deallocframe
  //   .Lend:
  uint32_t *FirstAddress = reinterpret_cast<uint32_t *>(Sled.address());
  if (Enable) {
    WriteInstFlushCache(FirstAddress, uint32_t(PO_NOP));
  } else {
    WriteInstFlushCache(FirstAddress, uint32_t(PO_JUMPI_30));
  }
  return false;
}

bool patchTypedEvent(const bool Enable, const uint32_t FuncId,
                     const XRaySledEntry &Sled) XRAY_NEVER_INSTRUMENT {
  // The typed event sled (3 args) is 15 words = 60 bytes:
  //   .Lxray_sled_N:
  //     { jump .Lend }     <-- first word: jump over (disabled) / nop (enabled)
  //     allocframe, sp adjust, 3 saves, 3 moves, call, 3 restores,
  //     sp adjust, deallocframe
  //   .Lend:
  uint32_t *FirstAddress = reinterpret_cast<uint32_t *>(Sled.address());
  if (Enable) {
    WriteInstFlushCache(FirstAddress, uint32_t(PO_NOP));
  } else {
    WriteInstFlushCache(FirstAddress, uint32_t(PO_JUMPI_3C));
  }
  return false;
}

} // namespace __xray

extern "C" void __xray_ArgLoggerEntry() XRAY_NEVER_INSTRUMENT {
  // FIXME: this will have to be implemented in the trampoline assembly file
}
