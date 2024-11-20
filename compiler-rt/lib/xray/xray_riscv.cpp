//===-- xray_riscv.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of XRay, a dynamic runtime instrumentation system.
//
// Implementation of riscv-specific routines (32- and 64-bit).
//
//===----------------------------------------------------------------------===//
#include "sanitizer_common/sanitizer_common.h"
#include "xray_defs.h"
#include "xray_interface_internal.h"
#include <atomic>

namespace __xray {

// The machine codes for some instructions used in runtime patching.
enum PatchOpcodes : uint32_t {
  PO_ADDI = 0x00000013, // addi rd, rs1, imm
  PO_ADD = 0x00000033,  // add rd, rs1, rs2
  PO_SW = 0x00002023,   // sw rt, base(offset)
  PO_SD = 0x00003023,   // sd rt, base(offset)
  PO_LUI = 0x00000037,  // lui rd, imm
  PO_ORI = 0x00006013,  // ori rd, rs1, imm
  PO_OR = 0x00006033,   // or rd, rs1, rs2
  PO_SLLI = 0x00001013, // slli rd, rs, shamt
  PO_SRLI = 0x00005013, // srli rd, rs, shamt
  PO_JALR = 0x00000067, // jalr rs
  PO_LW = 0x00002003,   // lw rd, base(offset)
  PO_LD = 0x00003003,   // ld rd, base(offset)
  PO_J = 0x0000006f,    // jal #n_bytes
  PO_NOP = 0x00000013,  // nop - pseduo-instruction, same as addi x0, x0, 0
};

enum RegNum : uint32_t {
  RN_R0 = 0x0,
  RN_RA = 0x1,
  RN_SP = 0x2,
  RN_T0 = 0x5,
  RN_T1 = 0x6,
  RN_T2 = 0x7,
  RN_A0 = 0xa,
};

static inline uint32_t encodeRTypeInstruction(uint32_t Opcode, uint32_t Rs1,
                                              uint32_t Rs2, uint32_t Rd) {
  return Rs2 << 20 | Rs1 << 15 | Rd << 7 | Opcode;
}

static inline uint32_t encodeITypeInstruction(uint32_t Opcode, uint32_t Rs1,
                                              uint32_t Rd, uint32_t Imm) {
  return Imm << 20 | Rs1 << 15 | Rd << 7 | Opcode;
}

static inline uint32_t encodeSTypeInstruction(uint32_t Opcode, uint32_t Rs1,
                                              uint32_t Rs2, uint32_t Imm) {
  uint32_t imm_msbs = (Imm & 0xfe0) << 25;
  uint32_t imm_lsbs = (Imm & 0x01f) << 7;
  return imm_msbs | Rs2 << 20 | Rs1 << 15 | imm_lsbs | Opcode;
}

static inline uint32_t encodeUTypeInstruction(uint32_t Opcode, uint32_t Rd,
                                              uint32_t Imm) {
  return Imm << 12 | Rd << 7 | Opcode;
}

static inline uint32_t encodeJTypeInstruction(uint32_t Opcode, uint32_t Rd,
                                              uint32_t Imm) {
  uint32_t imm_msb = (Imm & 0x80000) << 31;
  uint32_t imm_lsbs = (Imm & 0x003ff) << 21;
  uint32_t imm_11 = (Imm & 0x00400) << 20;
  uint32_t imm_1912 = (Imm & 0x7f800) << 12;
  return imm_msb | imm_lsbs | imm_11 | imm_1912 | Rd << 7 | Opcode;
}

#if SANITIZER_RISCV64
static uint32_t hi20(uint64_t val) { return (val + 0x800) >> 12; }
static uint32_t lo12(uint64_t val) { return val & 0xfff; }
#elif defined(__riscv) && (__riscv_xlen == 32)
static uint32_t hi20(uint32_t val) { return (val + 0x800) >> 12; }
static uint32_t lo12(uint32_t val) { return val & 0xfff; }
#endif

static inline bool patchSled(const bool Enable, const uint32_t FuncId,
                             const XRaySledEntry &Sled,
                             void (*TracingHook)()) XRAY_NEVER_INSTRUMENT {
  // When |Enable| == true,
  // We replace the following compile-time stub (sled):
  //
  // xray_sled_n:
  //	J .tmpN
  //	29 or 37 C.NOPs (58 or 74 bytes)
  //	.tmpN
  //
  // With one of the following runtime patches:
  //
  // xray_sled_n (32-bit):
  //    addi sp, sp, -16                                ;create stack frame
  //    sw ra, 12(sp)                                   ;save return address
  //    sw t2, 8(sp)                                    ;save register t2
  //    sw t1, 4(sp)                                    ;save register t1
  //    sw a0, 0(sp)                                    ;save register a0
  //    lui t1, %hi(__xray_FunctionEntry/Exit)
  //    addi t1, t1, %lo(__xray_FunctionEntry/Exit)
  //    lui a0, %hi(function_id)
  //    addi a0, a0, %lo(function_id)                   ;pass function id
  //    jalr t1                                         ;call Tracing hook
  //    lw a0, 0(sp)                                    ;restore register a0
  //    lw t1, 4(sp)                                    ;restore register t1
  //    lw t2, 8(sp)                                    ;restore register t2
  //    lw ra, 12(sp)                                   ;restore return address
  //    addi sp, sp, 16                                 ;delete stack frame
  //
  // xray_sled_n (64-bit):
  //    addi sp, sp, -32                                ;create stack frame
  //    sd ra, 24(sp)                                   ;save return address
  //    sd t2, 16(sp)                                   ;save register t2
  //    sd t1, 8(sp)                                    ;save register t1
  //    sd a0, 0(sp)                                    ;save register a0
  //    lui t2, %highest(__xray_FunctionEntry/Exit)
  //    addi t2, t2, %higher(__xray_FunctionEntry/Exit)
  //    slli t2, t2, 32
  //    lui t1, t1, %hi(__xray_FunctionEntry/Exit)
  //    addi t1, t1, %lo(__xray_FunctionEntry/Exit)
  //    add t1, t2, t1
  //    lui a0, %hi(function_id)
  //    addi a0, a0, %lo(function_id)                   ;pass function id
  //    jalr t1                                         ;call Tracing hook
  //    ld a0, 0(sp)                                    ;restore register a0
  //    ld t1, 8(sp)                                    ;restore register t1
  //    ld t2, 16(sp)                                   ;restore register t2
  //    ld ra, 24(sp)                                   ;restore return address
  //    addi sp, sp, 32                                 ;delete stack frame
  //
  // Replacement of the first 4-byte instruction should be the last and atomic
  // operation, so that the user code which reaches the sled concurrently
  // either jumps over the whole sled, or executes the whole sled when the
  // latter is ready.
  //
  // When |Enable|==false, we set back the first instruction in the sled to be
  //   J 60 bytes (rv32)
  //   J 76 bytes (rv64)

  uint32_t *Address = reinterpret_cast<uint32_t *>(Sled.address());
  if (Enable) {
    // If the ISA is RISCV 64, the Tracing Hook needs to be typecast to a 64 bit
    // value
#if SANITIZER_RISCV64
    uint32_t LoTracingHookAddr = lo12(reinterpret_cast<uint64_t>(TracingHook));
    uint32_t HiTracingHookAddr = hi20(reinterpret_cast<uint64_t>(TracingHook));
    uint32_t HigherTracingHookAddr =
        lo12((reinterpret_cast<uint64_t>(TracingHook) + 0x80000000) >> 32);
    uint32_t HighestTracingHookAddr =
        hi20((reinterpret_cast<uint64_t>(TracingHook) + 0x80000000) >> 32);
    // We typecast the Tracing Hook to a 32 bit value for RISCV32
#elif defined(__riscv) && (__riscv_xlen == 32)
    uint32_t LoTracingHookAddr = lo12(reinterpret_cast<uint32_t>(TracingHook));
    uint32_t HiTracingHookAddr = hi20((reinterpret_cast<uint32_t>(TracingHook));
#endif
    uint32_t LoFunctionID = lo12(FuncId);
    uint32_t HiFunctionID = hi20(FuncId);
    // The sled that is patched in for RISCV64 defined below. We need the entire
    // sleds corresponding to both ISAs to be protected by defines because the
    // first few instructions are all different, because we store doubles in
    // case of RV64 and store words for RV32. Subsequently, we have LUI - and in
    // case of RV64, we need extra instructions from this point on, so we see
    // differences in addresses to which instructions are stored.
#if SANITIZER_RISCV64
    Address[1] = encodeSTypeInstruction(PatchOpcodes::PO_SD, RegNum::RN_SP,
                                        RegNum::RN_RA, 0x18);
    Address[2] = encodeSTypeInstruction(PatchOpcodes::PO_SD, RegNum::RN_SP,
                                        RegNum::RN_T2, 0x10);
    Address[3] = encodeSTypeInstruction(PatchOpcodes::PO_SD, RegNum::RN_SP,
                                        RegNum::RN_T1, 0x8);
    Address[4] = encodeSTypeInstruction(PatchOpcodes::PO_SD, RegNum::RN_SP,
                                        RegNum::RN_A0, 0x0);
    Address[5] = encodeUTypeInstruction(PatchOpcodes::PO_LUI, RegNum::RN_T2,
                                        HighestTracingHookAddr);
    Address[6] = encodeITypeInstruction(PatchOpcodes::PO_ADDI, RegNum::RN_T2,
                                        RegNum::RN_T2, HigherTracingHookAddr);
    Address[7] = encodeITypeInstruction(PatchOpcodes::PO_SLLI, RegNum::RN_T2,
                                        RegNum::RN_T2, 0x20);
    Address[8] = encodeUTypeInstruction(PatchOpcodes::PO_LUI, RegNum::RN_T1,
                                        HiTracingHookAddr);
    Address[9] = encodeITypeInstruction(PatchOpcodes::PO_ADDI, RegNum::RN_T1,
                                        RegNum::RN_T1, LoTracingHookAddr);
    Address[10] = encodeRTypeInstruction(PatchOpcodes::PO_ADD, RegNum::RN_T1,
                                         RegNum::RN_T2, RegNum::RN_T1);
    Address[11] = encodeUTypeInstruction(PatchOpcodes::PO_LUI, RegNum::RN_A0,
                                         HiFunctionID);
    Address[12] = encodeITypeInstruction(PatchOpcodes::PO_ADDI, RegNum::RN_A0,
                                         RegNum::RN_A0, LoFunctionID);
    Address[13] = encodeITypeInstruction(PatchOpcodes::PO_JALR, RegNum::RN_T1,
                                         RegNum::RN_RA, 0x0);
    Address[14] = encodeITypeInstruction(PatchOpcodes::PO_LD, RegNum::RN_SP,
                                         RegNum::RN_A0, 0x0);
    Address[15] = encodeITypeInstruction(PatchOpcodes::PO_LD, RegNum::RN_SP,
                                         RegNum::RN_T1, 0x8);
    Address[16] = encodeITypeInstruction(PatchOpcodes::PO_LD, RegNum::RN_SP,
                                         RegNum::RN_T2, 0x10);
    Address[17] = encodeITypeInstruction(PatchOpcodes::PO_LD, RegNum::RN_SP,
                                         RegNum::RN_RA, 0x18);
    Address[18] = encodeITypeInstruction(PatchOpcodes::PO_ADDI, RegNum::RN_SP,
                                         RegNum::RN_SP, 0x20);
    uint32_t CreateStackSpace = encodeITypeInstruction(
        PatchOpcodes::PO_ADDI, RegNum::RN_SP, RegNum::RN_SP, 0xffe0);
#elif defined(__riscv) && (__riscv_xlen == 32)
    Address[1] = encodeSTypeInstruction(PatchOpcodes::PO_SW, RegNum::RN_SP,
                                        RegNum::RN_RA, 0x0c);
    Address[2] = encodeSTypeInstruction(PatchOpcodes::PO_SW, RegNum::RN_SP,
                                        RegNum::RN_T2, 0x08);
    Address[3] = encodeSTypeInstruction(PatchOpcodes::PO_SW, RegNum::RN_SP,
                                        RegNum::RN_T1, 0x4);
    Address[4] = encodeSTypeInstruction(PatchOpcodes::PO_SW, RegNum::RN_SP,
                                        RegNum::RN_A0, 0x0);
    Address[5] = encodeUTypeInstruction(PatchOpcodes::PO_LUI, RegNum::RN_T1,
                                        HiTracingHookAddr);
    Address[6] = encodeITypeInstruction(PatchOpcodes::PO_ADDI, RegNum::RN_T1,
                                        RegNum::RN_T1, LoTracingHookAddr);
    Address[7] = encodeUTypeInstruction(PatchOpcodes::PO_LUI, RegNum::RN_A0,
                                        HiFunctionID);
    Address[8] = encodeITypeInstruction(PatchOpcodes::PO_ADDI, RegNum::RN_A0,
                                        RegNum::RN_A0, LoFunctionID);
    Address[9] = encodeITypeInstruction(PatchOpcodes::PO_JALR, RegNum::RN_T1,
                                        RegNum::RN_RA, 0x0);
    Address[10] = encodeITypeInstruction(PatchOpcodes::PO_LW, RegNum::RN_SP,
                                         RegNum::RN_A0, 0x0);
    Address[11] = encodeITypeInstruction(PatchOpcodes::PO_LW, RegNum::RN_SP,
                                         RegNum::RN_T1, 0x4);
    Address[12] = encodeITypeInstruction(PatchOpcodes::PO_LW, RegNum::RN_SP,
                                         RegNum::RN_T2, 0x08);
    Address[13] = encodeITypeInstruction(PatchOpcodes::PO_LW, RegNum::RN_SP,
                                         RegNum::RN_RA, 0x0c);
    Address[14] = encodeITypeInstruction(PatchOpcodes::PO_ADDI, RegNum::RN_SP,
                                         RegNum::RN_SP, 0x10);
    uint32_t CreateStackSpace = encodeITypeInstruction(
        PatchOpcodes::PO_ADDI, RegNum::RN_SP, RegNum::RN_SP, 0xfff0);
#endif
    std::atomic_store_explicit(
        reinterpret_cast<std::atomic<uint32_t> *>(Address), CreateStackSpace,
        std::memory_order_release);
  } else {
    uint32_t CreateBranch = encodeJTypeInstruction(
    // Jump distance is different in both ISAs due to difference in size of
    // sleds
#if SANITIZER_RISCV64
        PatchOpcodes::PO_J, RegNum::RN_R0,
        0x026); // jump encodes an offset in multiples of 2 bytes. 38*2 = 76
#elif defined(__riscv) && (__riscv_xlen == 32)
        PatchOpcodes::PO_J, RegNum::RN_R0,
        0x01e); // jump encodes an offset in multiples of 2 bytes. 30*2 = 60
#endif
    std::atomic_store_explicit(
        reinterpret_cast<std::atomic<uint32_t> *>(Address), CreateBranch,
        std::memory_order_release);
  }
  return true;
}

bool patchFunctionEntry(const bool Enable, const uint32_t FuncId,
                        const XRaySledEntry &Sled,
                        const XRayTrampolines &Trampolines,
                        bool LogArgs) XRAY_NEVER_INSTRUMENT {
  // We don't support Logging argument at this moment, so we always
  // use EntryTrampoline.
  return patchSled(Enable, FuncId, Sled, Trampolines.EntryTrampoline);
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
  return false;
}

bool patchTypedEvent(const bool Enable, const uint32_t FuncId,
                     const XRaySledEntry &Sled) XRAY_NEVER_INSTRUMENT {
  return false;
}
} // namespace __xray

extern "C" void __xray_ArgLoggerEntry() XRAY_NEVER_INSTRUMENT {}
