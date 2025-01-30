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
// Implementation of RISC-V specific routines (32- and 64-bit).
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
  PO_SW = 0x00002023,   // sw rs2, imm(rs1)
  PO_SD = 0x00003023,   // sd rs2, imm(rs1)
  PO_LUI = 0x00000037,  // lui rd, imm
  PO_OR = 0x00006033,   // or rd, rs1, rs2
  PO_SLLI = 0x00001013, // slli rd, rs1, shamt
  PO_JALR = 0x00000067, // jalr rd, rs1
  PO_LW = 0x00002003,   // lw rd, imm(rs1)
  PO_LD = 0x00003003,   // ld rd, imm(rs1)
  PO_J = 0x0000006f,    // jal imm
  PO_NOP = PO_ADDI,     // addi x0, x0, 0
};

enum RegNum : uint32_t {
  RN_X0 = 0,
  RN_RA = 1,
  RN_SP = 2,
  RN_T1 = 6,
  RN_A0 = 10,
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
  uint32_t ImmMSB = (Imm & 0xfe0) << 20;
  uint32_t ImmLSB = (Imm & 0x01f) << 7;
  return ImmMSB | Rs2 << 20 | Rs1 << 15 | ImmLSB | Opcode;
}

static inline uint32_t encodeUTypeInstruction(uint32_t Opcode, uint32_t Rd,
                                              uint32_t Imm) {
  return Imm << 12 | Rd << 7 | Opcode;
}

static inline uint32_t encodeJTypeInstruction(uint32_t Opcode, uint32_t Rd,
                                              uint32_t Imm) {
  uint32_t ImmMSB = (Imm & 0x100000) << 11;
  uint32_t ImmLSB = (Imm & 0x7fe) << 20;
  uint32_t Imm11 = (Imm & 0x800) << 9;
  uint32_t Imm1912 = (Imm & 0xff000);
  return ImmMSB | ImmLSB | Imm11 | Imm1912 | Rd << 7 | Opcode;
}

static uint32_t hi20(uint32_t val) { return (val + 0x800) >> 12; }
static uint32_t lo12(uint32_t val) { return val & 0xfff; }

static inline bool patchSled(const bool Enable, const uint32_t FuncId,
                             const XRaySledEntry &Sled,
                             void (*TracingHook)()) XRAY_NEVER_INSTRUMENT {
  // When |Enable| == true,
  // We replace the following compile-time stub (sled):
  //
  // xray_sled_n:
  //	J .tmpN
  //	21 or 33 C.NOPs (42 or 66 bytes)
  //	.tmpN
  //
  // With one of the following runtime patches:
  //
  // xray_sled_n (32-bit):
  //    addi sp, sp, -16                                ;create stack frame
  //    sw ra, 12(sp)                                   ;save return address
  //    sw a0, 8(sp)                                    ;save register a0
  //    lui ra, %hi(__xray_FunctionEntry/Exit)
  //    addi ra, ra, %lo(__xray_FunctionEntry/Exit)
  //    lui a0, %hi(function_id)
  //    addi a0, a0, %lo(function_id)                   ;pass function id
  //    jalr ra                                         ;call Tracing hook
  //    lw a0, 8(sp)                                    ;restore register a0
  //    lw ra, 12(sp)                                   ;restore return address
  //    addi sp, sp, 16                                 ;delete stack frame
  //
  // xray_sled_n (64-bit):
  //    addi sp, sp, -32                                ;create stack frame
  //    sd ra, 24(sp)                                   ;save return address
  //    sd a0, 16(sp)                                   ;save register a0
  //    sd t1, 8(sp)                                    ;save register t1
  //    lui t1, %highest(__xray_FunctionEntry/Exit)
  //    addi t1, t1, %higher(__xray_FunctionEntry/Exit)
  //    slli t1, t1, 32
  //    lui ra, ra, %hi(__xray_FunctionEntry/Exit)
  //    addi ra, ra, %lo(__xray_FunctionEntry/Exit)
  //    add ra, t1, ra
  //    lui a0, %hi(function_id)
  //    addi a0, a0, %lo(function_id)                   ;pass function id
  //    jalr ra                                         ;call Tracing hook
  //    ld t1, 8(sp)                                    ;restore register t1
  //    ld a0, 16(sp)                                   ;restore register a0
  //    ld ra, 24(sp)                                   ;restore return address
  //    addi sp, sp, 32                                 ;delete stack frame
  //
  // Replacement of the first 4-byte instruction should be the last and atomic
  // operation, so that the user code which reaches the sled concurrently
  // either jumps over the whole sled, or executes the whole sled when the
  // latter is ready.
  //
  // When |Enable|==false, we set back the first instruction in the sled to be
  //   J 44 bytes (rv32)
  //   J 68 bytes (rv64)

  uint32_t *Address = reinterpret_cast<uint32_t *>(Sled.address());
  if (Enable) {
#if __riscv_xlen == 64
    // If the ISA is RV64, the Tracing Hook needs to be typecast to a 64 bit
    // value.
    uint32_t LoTracingHookAddr = lo12(reinterpret_cast<uint64_t>(TracingHook));
    uint32_t HiTracingHookAddr = hi20(reinterpret_cast<uint64_t>(TracingHook));
    uint32_t HigherTracingHookAddr =
        lo12((reinterpret_cast<uint64_t>(TracingHook) + 0x80000000) >> 32);
    uint32_t HighestTracingHookAddr =
        hi20((reinterpret_cast<uint64_t>(TracingHook) + 0x80000000) >> 32);
#elif __riscv_xlen == 32
    // We typecast the Tracing Hook to a 32 bit value for RV32
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
    size_t Idx = 1U;
    const uint32_t XLenBytes = __riscv_xlen / 8;
#if __riscv_xlen == 64
    const uint32_t LoadOp = PatchOpcodes::PO_LD;
    const uint32_t StoreOp = PatchOpcodes::PO_SD;
#elif __riscv_xlen == 32
    const uint32_t LoadOp = PatchOpcodes::PO_LW;
    const uint32_t StoreOp = PatchOpcodes::PO_SW;
#endif

    Address[Idx++] = encodeSTypeInstruction(StoreOp, RegNum::RN_SP,
                                            RegNum::RN_RA, 3 * XLenBytes);
    Address[Idx++] = encodeSTypeInstruction(StoreOp, RegNum::RN_SP,
                                            RegNum::RN_A0, 2 * XLenBytes);

#if __riscv_xlen == 64
    Address[Idx++] = encodeSTypeInstruction(StoreOp, RegNum::RN_SP,
                                            RegNum::RN_T1, XLenBytes);
    Address[Idx++] = encodeUTypeInstruction(PatchOpcodes::PO_LUI, RegNum::RN_T1,
                                            HighestTracingHookAddr);
    Address[Idx++] =
        encodeITypeInstruction(PatchOpcodes::PO_ADDI, RegNum::RN_T1,
                               RegNum::RN_T1, HigherTracingHookAddr);
    Address[Idx++] = encodeITypeInstruction(PatchOpcodes::PO_SLLI,
                                            RegNum::RN_T1, RegNum::RN_T1, 32);
#endif
    Address[Idx++] = encodeUTypeInstruction(PatchOpcodes::PO_LUI, RegNum::RN_RA,
                                            HiTracingHookAddr);
    Address[Idx++] = encodeITypeInstruction(
        PatchOpcodes::PO_ADDI, RegNum::RN_RA, RegNum::RN_RA, LoTracingHookAddr);
#if __riscv_xlen == 64
    Address[Idx++] = encodeRTypeInstruction(PatchOpcodes::PO_ADD, RegNum::RN_RA,
                                            RegNum::RN_T1, RegNum::RN_RA);
#endif
    Address[Idx++] = encodeUTypeInstruction(PatchOpcodes::PO_LUI, RegNum::RN_A0,
                                            HiFunctionID);
    Address[Idx++] = encodeITypeInstruction(
        PatchOpcodes::PO_ADDI, RegNum::RN_A0, RegNum::RN_A0, LoFunctionID);
    Address[Idx++] = encodeITypeInstruction(PatchOpcodes::PO_JALR,
                                            RegNum::RN_RA, RegNum::RN_RA, 0);

#if __riscv_xlen == 64
    Address[Idx++] =
        encodeITypeInstruction(LoadOp, RegNum::RN_SP, RegNum::RN_T1, XLenBytes);
#endif
    Address[Idx++] = encodeITypeInstruction(LoadOp, RegNum::RN_SP,
                                            RegNum::RN_A0, 2 * XLenBytes);
    Address[Idx++] = encodeITypeInstruction(LoadOp, RegNum::RN_SP,
                                            RegNum::RN_RA, 3 * XLenBytes);
    Address[Idx++] = encodeITypeInstruction(
        PatchOpcodes::PO_ADDI, RegNum::RN_SP, RegNum::RN_SP, 4 * XLenBytes);

    uint32_t CreateStackSpace = encodeITypeInstruction(
        PatchOpcodes::PO_ADDI, RegNum::RN_SP, RegNum::RN_SP, -4 * XLenBytes);

    std::atomic_store_explicit(
        reinterpret_cast<std::atomic<uint32_t> *>(Address), CreateStackSpace,
        std::memory_order_release);
  } else {
    uint32_t CreateBranch = encodeJTypeInstruction(
    // Jump distance is different in both ISAs due to difference in size of
    // sleds
#if __riscv_xlen == 64
        PatchOpcodes::PO_J, RegNum::RN_X0,
        68); // jump encodes an offset of 68
#elif __riscv_xlen == 32
        PatchOpcodes::PO_J, RegNum::RN_X0,
        44); // jump encodes an offset of 44
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
  // We don't support logging argument at this moment, so we always
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
