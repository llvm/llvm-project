//===- comgr-hotswap-patch-f32-to-e5m3.cpp - E5M3 CLAMP-bit emulation ----===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Scratch-patch pass for Case 1 of the B0-to-A0 scratch-patch pipeline:
/// E5M3 CLAMP-bit emulation for FP8 conversion instructions.
///
/// GFX1250 B0 added a CLAMP bit to three FP8 conversion instructions that
/// selects UE5M3 format (CLAMP=1) instead of E4M3 (CLAMP=0). On A0 the
/// CLAMP bit is non-functional: CLAMP=1 silently produces E4M3. This file
/// provides the strong override of applyScratchPatches that detects CLAMP=1
/// FP8 conversions and emits software emulation sequences.
///
/// Covered instructions:
///   1. v_cvt_pk_fp8_f32  - F32 pack to FP8
///   2. v_cvt_sr_fp8_f32  - F32 stochastic-round to FP8
///   3. v_cvt_f32_fp8     - FP8 unpack to F32
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"

using namespace llvm;

namespace COMGR {
namespace hotswap {

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

std::string vgprName(unsigned N) { return ("v" + Twine(N)).str(); }
std::string sgprName(unsigned N) { return ("s" + Twine(N)).str(); }

/// Convert an MCRegister to its assembly name (e.g. VGPR0 to "v0").
/// The MCRegisterInfo name is the tablegen C identifier (e.g. "VGPR0",
/// "SGPR0", "VCC_LO"); we map it to the assembler name used in inline asm.
/// True16 sub-register names ("VGPR0_LO16") are mapped to the 32-bit
/// parent since the replacement sequences only use 32-bit ALU instructions.
// TODO: Extract toAsmRegName into comgr-hotswap-internal.h as a shared
// utility. A similar version exists in comgr-hotswap-patch-trampoline.cpp.
// This one additionally handles True16 suffixes. Tracked in #2253.
std::string toAsmRegName(const MCRegisterInfo &MRI, MCRegister Reg) {
  const char *N = MRI.getName(Reg);
  if (!N)
    return {};
  StringRef S(N);
  // True16 sub-registers: VGPR0_LO16 / VGPR0_HI16 use 32-bit parent.
  if (S.contains("_LO16") || S.contains("_HI16"))
    S = S.take_until([](char C) { return C == '_'; });
  if (S.consume_front("VGPR"))
    return ("v" + S).str();
  if (S.consume_front("SGPR"))
    return ("s" + S).str();
  return S.lower();
}

/// Format a source operand with its SISrcMods modifier flags for use in
/// floating-point ALU instructions (e.g. v_max_num_f32).
/// SISrcMods: NEG=1, ABS=2.
std::string fmtModifiedSrc(StringRef BareReg, unsigned Mods) {
  bool Neg = Mods & 1;
  bool Abs = Mods & 2;
  std::string R = BareReg.str();
  if (Abs)
    R = "abs(" + R + ")";
  if (Neg)
    R = "-" + R;
  return R;
}

std::string fmtLiteral32(int64_t Imm) {
  return "0x" + utohexstr(static_cast<uint32_t>(Imm));
}

bool isRegOrImm(const MCOperand &Op) { return Op.isReg() || Op.isImm(); }

/// Format a register or literal F32 source for replacement assembly. Literal
/// VOP3 operands are materialized into \p TempReg so the emulation sequence can
/// freely use integer ALU instructions that otherwise cannot carry multiple
/// literal operands.
bool materializeF32Source(raw_string_ostream &OS, const MCOperand &Op,
                          unsigned Mods, StringRef TempReg,
                          const MCRegisterInfo &MRI, uint64_t Offset,
                          StringRef InstName, std::string &BareSrc,
                          std::string &ModifiedSrc) {
  if (Op.isReg()) {
    BareSrc = toAsmRegName(MRI, Op.getReg());
    ModifiedSrc = fmtModifiedSrc(BareSrc, Mods);
    return true;
  }

  if (Op.isImm()) {
    BareSrc = TempReg.str();
    OS << "v_mov_b32 " << BareSrc << ", " << fmtLiteral32(Op.getImm()) << "\n";
    ModifiedSrc = fmtModifiedSrc(BareSrc, Mods);
    return true;
  }

  log() << "hotswap: error: " << InstName
        << ": unsupported source operand at 0x" << utohexstr(Offset) << "\n";
  return false;
}

// -- VOP3 operand layout structs -------------------------------------------
//
// Mirrors the MCInst operand order produced by the AMDGPU disassembler for
// the FP8 conversion instructions.  The indices below are validated at
// runtime in each patch function; a mismatch logs an error and bails out
// rather than silently reading the wrong operand.  This follows the same
// pattern as VOP3PWmmaLayout in comgr-hotswap-patch-wmma-split.cpp.

// VOP3 MCInst layout for the two-source FP8 pack / stochastic-round
// instructions.  op_sel and byte_sel are NOT separate MCInst operands; they
// are folded into the src-modifier immediates by the AMDGPU disassembler.
//
// MCInst operand order (verified at runtime):
//   [0] vdst       (reg)
//   [1] src0_mods  (imm - SISrcMods flags including OP_SEL bits)
//   [2] src0       (reg)
//   [3] src1_mods  (imm)
//   [4] src1       (reg)
//   [5] clamp      (imm)
//   [6] vdst_in    (reg - tied to vdst)
struct VOP3Fp8TwoSrcLayout {
  unsigned NumOperands;
  unsigned VDst;
  unsigned Src0Mods;
  unsigned Src0;
  unsigned Src1Mods;
  unsigned Src1;
  unsigned Clamp;
  unsigned VDstIn;
};

// v_cvt_f32_fp8 (gfx1250, VOP3 unpack - no source modifiers):
//   [0] vdst       (reg)
//   [1] src0       (reg)
//   [2] clamp      (imm)
//   [3] byte_sel   (imm)
struct VOP3Fp8UnpackLayout {
  unsigned NumOperands;
  unsigned VDst;
  unsigned Src0;
  unsigned Clamp;
  unsigned ByteSel;
};

constexpr VOP3Fp8TwoSrcLayout TwoSrcFp8Layout = {
    /*NumOperands=*/7, /*VDst=*/0,
    /*Src0Mods=*/1,    /*Src0=*/2,
    /*Src1Mods=*/3,    /*Src1=*/4,
    /*Clamp=*/5,       /*VDstIn=*/6};

constexpr VOP3Fp8UnpackLayout UnpackFp8Layout = {/*NumOperands=*/4, /*VDst=*/0,
                                                 /*Src0=*/1, /*Clamp=*/2,
                                                 /*ByteSel=*/3};

// -- Scratch allocation helper -----------------------------------------------

struct ScratchAllocation {
  VgprAllocator VgprAlloc;
  SgprAllocator SgprAlloc;
  std::string KernelName;
};

ScratchAllocation allocateScratch(PatchContext &Ctx, size_t Idx) {
  const InternalDecodedInst &DI = Ctx.Decoded[Idx];
  std::string KernelName =
      Ctx.Elf.findKernelAtAddress(DI.Offset + Ctx.Elf.textAddr());
  std::optional<unsigned> KdVgprs = Ctx.Elf.getKernelVgprCount(
      KernelName, getKernelVgprGranuleSize(Ctx, KernelName));
  unsigned VgprKdCount = KdVgprs.value_or(Ctx.Config.MaxVgprs);
  VgprAllocator VgprAlloc(Ctx.Liveness.LiveBefore[Idx], VgprKdCount,
                          Ctx.Config.MaxVgprs);

  std::optional<unsigned> KdSgprs = Ctx.Elf.getKernelSgprCount(KernelName);
  unsigned SgprKdCount = KdSgprs.value_or(Ctx.Config.MaxSgprs);
  SgprAllocator SgprAlloc(SgprKdCount, Ctx.Config.MaxSgprs);

  return ScratchAllocation{std::move(VgprAlloc), std::move(SgprAlloc),
                           std::move(KernelName)};
}

// ---------------------------------------------------------------------------
// Per-source F32 to UE5M3 conversion with full fixups
// ---------------------------------------------------------------------------

/// Emit assembly for converting one F32 source to a UE5M3 byte in \p Out.
///
/// Handles NaN (to 0xFF), overflow/Inf (to 0xFE), RTE rounding of the 7
/// discarded F16 mantissa bits, and source modifiers (neg/abs forwarded
/// from the original instruction via \p Src).
///
/// Register contract:
///   \p Out - output VGPR, receives UE5M3 byte in bits [7:0]
///   \p Tmp - scratch VGPR, clobbered
///   \p TopByte - scratch VGPR, clobbered
///   \p NanSgpr - SGPR name (e.g. "s0") for saving/restoring the NaN flag
///   \p TopSgpr - SGPR name for saving/restoring the high-range flag
///   \p Src - full operand with modifiers (used in v_max_num_f32)
///   \p BareSrc - bare register name (used in v_and_b32 for NaN detect)
///   VCC is clobbered.
///
/// RTE rounding shortcut: rather than extracting round_bit, sticky, and lsb
/// into separate low-path registers, we use the identity:
/// round_up = (guard_bits * 2 + lsb) > 128, where
/// guard_bits = F16[6:0].  This collapses the entire RTE decision into a
/// single integer comparison via v_bfi_b32 + v_cmp + v_add_co_ci_u32.
void emitF32ToUE5M3(raw_string_ostream &OS, StringRef Src, StringRef BareSrc,
                    StringRef Out, StringRef Tmp, StringRef TopByte,
                    StringRef NanSgpr, StringRef TopSgpr) {
  // NaN detection: (|src| > 0x7F800000) means NaN.
  // v_and_b32 strips the sign, so neg/abs modifiers don't affect this test.
  // VOPC form: literal in src0, VGPR in src1 (implicit VCC write).
  OS << "v_and_b32 " << Tmp << ", 0x7FFFFFFF, " << BareSrc << "\n";
  OS << "v_cmp_lt_u32 0x7F800000, " << Tmp << "\n";
  OS << "s_mov_b32 " << NanSgpr << ", vcc_lo\n";

  // Clamp to non-negative. Source modifiers are applied by v_max_num_f32.
  OS << "v_max_num_f32 " << Out << ", 0, " << Src << "\n";

  // UE5M3 values 0xF8..0xFE represent the exponent-31 finite range that an
  // F16 intermediate cannot encode. For the PK instruction's RTE mode, select
  // this direct path at the 0xF7/0xF8 midpoint (63488.0) and compute:
  //   byte = min(0xFE, 0xF0 + round_nearest_even(src / 8192.0)).
  // The low-range F16 path below is still used for all smaller values.
  OS << "v_cmp_le_f32 0x47780000, " << Out << "\n";
  OS << "s_mov_b32 " << TopSgpr << ", vcc_lo\n";
  OS << "v_mul_f32 " << TopByte << ", 0x39000000, " << Out << "\n";
  OS << "v_cvt_nearest_i32_f32 " << TopByte << ", " << TopByte << "\n";
  OS << "v_add_u32 " << TopByte << ", 0xF0, " << TopByte << "\n";
  OS << "v_min_u32 " << TopByte << ", 0xFE, " << TopByte << "\n";

  OS << "v_cvt_f16_f32 " << Out << ", " << Out << "\n";

  // RTE rounding: extract guard_bits = F16[6:0], shift to get preliminary
  // byte, then compute round_up = (guard_bits*2 + lsb) > 128.
  // GFX1250 encodes v_cvt_f16_f32 as a True16 low-half write, so use a
  // bitfield extract for F16[15:7] instead of a 32-bit shift that would pull
  // preserved high-half bits into the FP8 byte.
  OS << "v_and_b32 " << Tmp << ", 0x7F, " << Out << "\n";
  OS << "v_bfe_u32 " << Out << ", " << Out << ", 7, 9\n";
  OS << "v_lshlrev_b32 " << Tmp << ", 1, " << Tmp << "\n";
  // v_bfi_b32: dst = (mask & insert) | (~mask & background)
  // With mask=0xFFFFFFFE: copies Tmp[31:1] and Out[0] to guard*2 + lsb.
  OS << "v_bfi_b32 " << Tmp << ", 0xFFFFFFFE, " << Tmp << ", " << Out << "\n";
  // v_add_co_ci_u32 adds VCC as carry-in, collapsing the conditional
  // increment into one instruction: Out += (guard*2 + lsb > 128) ? 1 : 0.
  OS << "v_cmp_lt_u32 0x80, " << Tmp << "\n";
  OS << "v_add_co_ci_u32 " << Out << ", 0, " << Out << "\n";

  // Select the direct exponent-31 result when the source is in the UE5M3
  // high range. The min remains as a defensive cap for the low path's carry
  // out and for any saturated direct-path integer conversion.
  OS << "v_min_u32 " << Out << ", 0xFE, " << Out << "\n";
  OS << "s_mov_b32 vcc_lo, " << TopSgpr << "\n";
  OS << "v_cndmask_b32 " << Out << ", " << Out << ", " << TopByte << "\n";

  // NaN override: if original F32 was NaN, force 0xFF.
  // Load the NaN byte into Tmp (avoids literal in v_cndmask src1).
  OS << "s_mov_b32 vcc_lo, " << NanSgpr << "\n";
  OS << "v_mov_b32 " << Tmp << ", 0xFF\n";
  OS << "v_cndmask_b32 " << Out << ", " << Out << ", " << Tmp << "\n";
}

// ---------------------------------------------------------------------------
// v_cvt_pk_fp8_f32 patch  (Case 1, instruction 1)
// ---------------------------------------------------------------------------

uint32_t patchCvtPkFp8F32(PatchContext &Ctx, size_t Idx) {
  const InternalDecodedInst &DI = Ctx.Decoded[Idx];
  const MCInst &Inst = DI.Inst;
  const VOP3Fp8TwoSrcLayout &L = TwoSrcFp8Layout;

  if (Inst.getNumOperands() < L.NumOperands) {
    log() << "hotswap: error: cvt_pk_fp8_f32: operand count mismatch: "
          << "expected " << L.NumOperands << ", got " << Inst.getNumOperands()
          << " at offset 0x" << utohexstr(DI.Offset) << "\n";
    return 0;
  }

  if (!Inst.getOperand(L.Clamp).isImm() ||
      Inst.getOperand(L.Clamp).getImm() == 0)
    return 0;

  if (DI.Size != 8 && DI.Size != 12) {
    log() << "hotswap: error: cvt_pk_fp8_f32: unexpected inst size " << DI.Size
          << " at offset 0x" << utohexstr(DI.Offset) << "\n";
    return 0;
  }

  // OPSEL[3] (write-high) is folded into src0_mods by the disassembler.
  // SISrcMods encodes OP_SEL_1 at bit 3 (value 8).
  unsigned Src0Mods = Inst.getOperand(L.Src0Mods).isImm()
                          ? Inst.getOperand(L.Src0Mods).getImm()
                          : 0;
  bool WriteHigh = (Src0Mods >> 3) & 1;

  const MCRegisterInfo &MRI = *Ctx.LS.MRI;
  if (!Inst.getOperand(L.VDst).isReg() ||
      !isRegOrImm(Inst.getOperand(L.Src0)) ||
      !isRegOrImm(Inst.getOperand(L.Src1))) {
    log() << "hotswap: error: cvt_pk_fp8_f32: unexpected imm operand at 0x"
          << utohexstr(DI.Offset) << "\n";
    return 0;
  }
  std::string VdstStr = toAsmRegName(MRI, Inst.getOperand(L.VDst).getReg());
  unsigned Src1Mods = Inst.getOperand(L.Src1Mods).isImm()
                          ? Inst.getOperand(L.Src1Mods).getImm()
                          : 0;

  ScratchAllocation SA = allocateScratch(Ctx, Idx);

  // 4 scratch VGPRs: T0 (src0 byte), T1 (src1 byte), T2 (shared scratch
  // for NaN detection and RTE rounding intermediates), T3 (high-range byte).
  std::optional<unsigned> T0 = SA.VgprAlloc.alloc();
  std::optional<unsigned> T1 = SA.VgprAlloc.alloc();
  std::optional<unsigned> T2 = SA.VgprAlloc.alloc();
  std::optional<unsigned> T3 = SA.VgprAlloc.alloc();
  if (!T0 || !T1 || !T2 || !T3) {
    log() << "hotswap: error: cvt_pk_fp8_f32: unable to allocate 4 scratch "
          << "VGPRs at offset 0x" << utohexstr(DI.Offset) << "\n";
    return 0;
  }

  std::string T0Name = vgprName(*T0);
  std::string T1Name = vgprName(*T1);
  std::string T2Name = vgprName(*T2);
  std::string T3Name = vgprName(*T3);

  std::optional<unsigned> NaN0Sgpr = SA.SgprAlloc.alloc();
  std::optional<unsigned> NaN1Sgpr = SA.SgprAlloc.alloc();
  std::optional<unsigned> TopSgpr = SA.SgprAlloc.alloc();
  std::optional<unsigned> VccSaveSgpr = SA.SgprAlloc.alloc();
  if (!NaN0Sgpr || !NaN1Sgpr || !TopSgpr || !VccSaveSgpr) {
    log() << "hotswap: error: cvt_pk_fp8_f32: unable to allocate 4 scratch "
          << "SGPRs at offset 0x" << utohexstr(DI.Offset) << "\n";
    return 0;
  }

  std::string NaN0Name = sgprName(*NaN0Sgpr);
  std::string NaN1Name = sgprName(*NaN1Sgpr);
  std::string TopName = sgprName(*TopSgpr);
  std::string VccSaveName = sgprName(*VccSaveSgpr);

  std::string Asm;
  raw_string_ostream AsmOS(Asm);

  // Save VCC before clobbering it with v_cmp_* instructions.
  AsmOS << "s_mov_b32 " << VccSaveName << ", vcc_lo\n";

  std::string Src0Bare;
  std::string Src1Bare;
  std::string Src0Str;
  std::string Src1Str;
  if (!materializeF32Source(AsmOS, Inst.getOperand(L.Src0), Src0Mods, T0Name,
                            MRI, DI.Offset, "cvt_pk_fp8_f32", Src0Bare,
                            Src0Str) ||
      !materializeF32Source(AsmOS, Inst.getOperand(L.Src1), Src1Mods, T1Name,
                            MRI, DI.Offset, "cvt_pk_fp8_f32", Src1Bare,
                            Src1Str))
    return 0;

  // --- src0 to byte in T0 (scratch: T2) ---
  emitF32ToUE5M3(AsmOS, Src0Str, Src0Bare, T0Name, T2Name, T3Name, NaN0Name,
                 TopName);

  // --- src1 to byte in T1 (scratch: T2) ---
  emitF32ToUE5M3(AsmOS, Src1Str, Src1Bare, T1Name, T2Name, T3Name, NaN1Name,
                 TopName);

  // Pack: T0[15:0] = { byte1, byte0 }
  AsmOS << "v_lshl_or_b32 " << T0Name << ", " << T1Name << ", 8, " << T0Name
        << "\n";

  // Merge into the correct 16-bit half of vdst.
  if (!WriteHigh) {
    AsmOS << "v_bfi_b32 " << VdstStr << ", 0xFFFF, " << T0Name << ", "
          << VdstStr << "\n";
  } else {
    AsmOS << "v_lshlrev_b32 " << T0Name << ", 16, " << T0Name << "\n";
    AsmOS << "v_bfi_b32 " << VdstStr << ", 0xFFFF0000, " << T0Name << ", "
          << VdstStr << "\n";
  }

  // Restore VCC to its pre-patch value.
  AsmOS << "s_mov_b32 vcc_lo, " << VccSaveName << "\n";

  SmallVector<uint8_t> ReplacementBytes = assembleInstructions(Asm, Ctx.LS);
  if (ReplacementBytes.empty()) {
    log() << "hotswap: error: cvt_pk_fp8_f32: assembly failed for "
          << "replacement at offset 0x" << utohexstr(DI.Offset) << "\n";
    return 0;
  }

  unsigned ExtraV = SA.VgprAlloc.extraVgprsNeeded();
  if (checkKernelVgprBump(Ctx, SA.KernelName, ExtraV,
                          PatchRequirement::Optional) !=
      VgprBumpDecision::Apply)
    return 0;

  if (!emitReplacementCode(Ctx, DI.Offset, DI.Size, ReplacementBytes))
    return 0;

  if (!SA.KernelName.empty()) {
    KernelPatchStats &Stats = Ctx.KernelStats[SA.KernelName];
    Stats.ExtraVgprs = std::max(Stats.ExtraVgprs, ExtraV);
    Stats.ExtraSgprs =
        std::max(Stats.ExtraSgprs, SA.SgprAlloc.extraSgprsNeeded());
    Stats.ScratchReused += 4 - ExtraV;
    Stats.ScratchAboveKd += ExtraV;
  }

  ScratchPatchInfo Info;
  Info.Offset = DI.Offset;
  Info.ScratchRegs = SA.VgprAlloc.LiveAtPoint;
  Ctx.OutScratchPatches.push_back(std::move(Info));

  log() << "hotswap: cvt_pk_fp8_f32: patched CLAMP=1 (E5M3) at offset 0x"
        << utohexstr(DI.Offset) << " (" << ReplacementBytes.size()
        << " bytes, scratch v" << *T0 << "/v" << *T1 << "/v" << *T2 << "/v"
        << *T3 << ", half=" << (WriteHigh ? "high" : "low") << ")\n";

  return 1;
}

// ---------------------------------------------------------------------------
// v_cvt_sr_fp8_f32 patch  (Case 1, instruction 2)
// ---------------------------------------------------------------------------

/// Patch a CLAMP=1 `v_cvt_sr_fp8_f32` (stochastic-round F32 to UE5M3).
///
/// The SR path injects stochastic noise into the F32 mantissa before the
/// F16 intermediate conversion, replicating the ISA pseudocode (17.6.94).
/// Unlike the PK path, no explicit RTE rounding block is needed; the
/// stochastic noise makes simple truncation statistically equivalent to
/// unbiased rounding (the noise carry already provides the correct
/// rounding probability without guard-bit extraction).
///
/// Scratch: 3 VGPRs (Out + Tmp + high-range byte), 3 SGPRs (NaN flag,
/// high-range flag, and VCC save).
uint32_t patchCvtSrFp8F32(PatchContext &Ctx, size_t Idx) {
  const InternalDecodedInst &DI = Ctx.Decoded[Idx];
  if (DI.Size != 8) {
    log() << "hotswap: error: cvt_sr_fp8_f32: unexpected inst size " << DI.Size
          << " at offset 0x" << utohexstr(DI.Offset) << "\n";
    return 0;
  }

  const MCInst &Inst = DI.Inst;
  const VOP3Fp8TwoSrcLayout &L = TwoSrcFp8Layout;

  if (Inst.getNumOperands() < L.NumOperands) {
    log() << "hotswap: error: cvt_sr_fp8_f32: operand count mismatch: "
          << "expected " << L.NumOperands << ", got " << Inst.getNumOperands()
          << " at offset 0x" << utohexstr(DI.Offset) << "\n";
    return 0;
  }

  if (!Inst.getOperand(L.Clamp).isImm() ||
      Inst.getOperand(L.Clamp).getImm() == 0)
    return 0;

  // byte_sel = OPSEL[3:2] in the VOP3 dword-0 encoding (bits [13:12]).
  // Unlike WriteHigh (OPSEL[3] at SISrcMods bit 3), the disassembler does
  // NOT fold byte_sel's OPSEL[3:2] into src0_mods for v_cvt_sr_fp8_f32:
  // they are only accessible from the raw encoding.
  const uint8_t *Raw = Ctx.Text + DI.Offset;
  unsigned ByteSel = (Raw[1] >> 5) & 0x3;

  unsigned Src0Mods = Inst.getOperand(L.Src0Mods).isImm()
                          ? Inst.getOperand(L.Src0Mods).getImm()
                          : 0;

  const MCRegisterInfo &MRI = *Ctx.LS.MRI;
  if (!Inst.getOperand(L.VDst).isReg() || !Inst.getOperand(L.Src0).isReg() ||
      !Inst.getOperand(L.Src1).isReg()) {
    log() << "hotswap: error: cvt_sr_fp8_f32: unexpected imm operand at 0x"
          << utohexstr(DI.Offset) << "\n";
    return 0;
  }
  std::string VdstStr = toAsmRegName(MRI, Inst.getOperand(L.VDst).getReg());
  std::string Src0Bare = toAsmRegName(MRI, Inst.getOperand(L.Src0).getReg());
  // src1 is U32 stochastic noise; ISA does not define NEG/ABS on integer
  // operands (OPF_NEG_1/OPF_ABS_1 absent for this opcode), so we use the
  // bare register name without modifier wrapping.
  std::string Src1Str = toAsmRegName(MRI, Inst.getOperand(L.Src1).getReg());
  std::string Src0Str = fmtModifiedSrc(Src0Bare, Src0Mods);

  ScratchAllocation SA = allocateScratch(Ctx, Idx);

  // 3 scratch VGPRs: Out (conversion result + noise intermediate), Tmp (NaN
  // flag save + noise computation), TopByte (high-range byte).
  std::optional<unsigned> Out = SA.VgprAlloc.alloc();
  std::optional<unsigned> Tmp = SA.VgprAlloc.alloc();
  std::optional<unsigned> TopByte = SA.VgprAlloc.alloc();
  if (!Out || !Tmp || !TopByte) {
    log() << "hotswap: error: cvt_sr_fp8_f32: unable to allocate 3 scratch "
          << "VGPRs at offset 0x" << utohexstr(DI.Offset) << "\n";
    return 0;
  }

  std::string OutName = vgprName(*Out);
  std::string TmpName = vgprName(*Tmp);
  std::string TopByteName = vgprName(*TopByte);

  std::optional<unsigned> NaNSgpr = SA.SgprAlloc.alloc();
  std::optional<unsigned> TopSgpr = SA.SgprAlloc.alloc();
  std::optional<unsigned> VccSaveSgpr = SA.SgprAlloc.alloc();
  if (!NaNSgpr || !TopSgpr || !VccSaveSgpr) {
    log() << "hotswap: error: cvt_sr_fp8_f32: unable to allocate 3 scratch "
          << "SGPRs at offset 0x" << utohexstr(DI.Offset) << "\n";
    return 0;
  }

  std::string NaNName = sgprName(*NaNSgpr);
  std::string TopName = sgprName(*TopSgpr);
  std::string VccSaveName = sgprName(*VccSaveSgpr);

  std::string Asm;
  raw_string_ostream AsmOS(Asm);

  // Save VCC before clobbering it with v_cmp_* instructions.
  AsmOS << "s_mov_b32 " << VccSaveName << ", vcc_lo\n";

  // --- NaN detection (before max destroys NaN) ---
  // v_and_b32 strips the sign, making this modifier-agnostic.
  AsmOS << "v_and_b32 " << TmpName << ", 0x7FFFFFFF, " << Src0Bare << "\n";
  AsmOS << "v_cmp_lt_u32 0x7F800000, " << TmpName << "\n";
  AsmOS << "s_mov_b32 " << NaNName << ", vcc_lo\n";

  // --- Clamp negative (UE5M3 is unsigned) ---
  // Source modifiers on src0 are applied natively by v_max_num_f32 VOP3.
  AsmOS << "v_max_num_f32 " << OutName << ", 0, " << Src0Str << "\n";

  // --- Stochastic noise injection ---
  // Add S1[31:12] as noise directly to the F32 bitpattern; carry
  // propagates into the exponent naturally.
  AsmOS << "v_lshrrev_b32 " << TmpName << ", 12, " << Src1Str << "\n";
  AsmOS << "v_add_u32 " << OutName << ", " << OutName << ", " << TmpName
        << "\n";

  // --- High-range direct path ---
  // SR mode injects rounding noise before truncation, so exponent-31 values
  // use floor(src / 8192.0) rather than the PK path's nearest conversion.
  AsmOS << "v_cmp_le_f32 0x47800000, " << OutName << "\n";
  AsmOS << "s_mov_b32 " << TopName << ", vcc_lo\n";
  AsmOS << "v_mul_f32 " << TopByteName << ", 0x39000000, " << OutName << "\n";
  AsmOS << "v_cvt_floor_i32_f32 " << TopByteName << ", " << TopByteName << "\n";
  AsmOS << "v_add_u32 " << TopByteName << ", 0xF0, " << TopByteName << "\n";
  AsmOS << "v_min_u32 " << TopByteName << ", 0xFE, " << TopByteName << "\n";

  // --- F32 to F16 to UE5M3 ---
  // Truncation is used here; SR noise handles rounding.
  AsmOS << "v_cvt_f16_f32 " << OutName << ", " << OutName << "\n";
  AsmOS << "v_bfe_u32 " << OutName << ", " << OutName << ", 7, 9\n";

  // --- Overflow clamp and high-range select ---
  AsmOS << "v_min_u32 " << OutName << ", 0xFE, " << OutName << "\n";
  AsmOS << "s_mov_b32 vcc_lo, " << TopName << "\n";
  AsmOS << "v_cndmask_b32 " << OutName << ", " << OutName << ", " << TopByteName
        << "\n";

  // --- NaN override ---
  AsmOS << "s_mov_b32 vcc_lo, " << NaNName << "\n";
  AsmOS << "v_mov_b32 " << TmpName << ", 0xFF\n";
  AsmOS << "v_cndmask_b32 " << OutName << ", " << OutName << ", " << TmpName
        << "\n";

  // --- Byte merge (byte_sel known at patch time) ---
  if (ByteSel == 0) {
    AsmOS << "v_bfi_b32 " << VdstStr << ", 0xFF, " << OutName << ", " << VdstStr
          << "\n";
  } else {
    unsigned Shift = ByteSel * 8;
    static const char *const Masks[] = {nullptr, "0xFF00", "0xFF0000",
                                        "0xFF000000"};
    AsmOS << "v_lshlrev_b32 " << OutName << ", " << Shift << ", " << OutName
          << "\n";
    AsmOS << "v_bfi_b32 " << VdstStr << ", " << Masks[ByteSel] << ", "
          << OutName << ", " << VdstStr << "\n";
  }

  // Restore VCC to its pre-patch value.
  AsmOS << "s_mov_b32 vcc_lo, " << VccSaveName << "\n";

  SmallVector<uint8_t> ReplacementBytes = assembleInstructions(Asm, Ctx.LS);
  if (ReplacementBytes.empty()) {
    log() << "hotswap: error: cvt_sr_fp8_f32: assembly failed for "
          << "replacement at offset 0x" << utohexstr(DI.Offset) << "\n";
    return 0;
  }

  unsigned ExtraV = SA.VgprAlloc.extraVgprsNeeded();
  if (checkKernelVgprBump(Ctx, SA.KernelName, ExtraV,
                          PatchRequirement::Optional) !=
      VgprBumpDecision::Apply)
    return 0;

  if (!emitToTrampoline(Ctx, DI.Offset, DI.Size, ReplacementBytes))
    return 0;

  if (!SA.KernelName.empty()) {
    KernelPatchStats &Stats = Ctx.KernelStats[SA.KernelName];
    Stats.ExtraVgprs = std::max(Stats.ExtraVgprs, ExtraV);
    Stats.ExtraSgprs =
        std::max(Stats.ExtraSgprs, SA.SgprAlloc.extraSgprsNeeded());
    Stats.ScratchReused += 3 - ExtraV;
    Stats.ScratchAboveKd += ExtraV;
  }

  ScratchPatchInfo Info;
  Info.Offset = DI.Offset;
  Info.ScratchRegs = SA.VgprAlloc.LiveAtPoint;
  Ctx.OutScratchPatches.push_back(std::move(Info));

  log() << "hotswap: cvt_sr_fp8_f32: patched CLAMP=1 (E5M3) at offset 0x"
        << utohexstr(DI.Offset) << " (" << ReplacementBytes.size()
        << " bytes, scratch v" << *Out << "/v" << *Tmp << "/v" << *TopByte
        << ", byte_sel=" << ByteSel << ")\n";

  return 1;
}

// ---------------------------------------------------------------------------
// v_cvt_f32_fp8 patch  (Case 1, instruction 3)
// ---------------------------------------------------------------------------

/// Patch a CLAMP=1 `v_cvt_f32_fp8` (UE5M3 to F32 unpack).
///
/// The unpack path extracts a UE5M3 byte from the source VGPR (position
/// selected by byte_sel), converts it to F32 via a left-shift-7 to F16 to
/// F32 pipeline, and applies fixups for the exponent-31 octave (bytes
/// 0xF8-0xFE, which the F16 path maps to +Inf instead of finite values)
/// and UE5M3 NaN (byte 0xFF).
///
/// Only VOP3 (_e64) encoding can carry CLAMP=1; VOP1 has no CLAMP bit and
/// is skipped.  No source modifiers exist on this instruction (OPF_NOABS,
/// OPF_NONEG) so no modifier forwarding is needed.
///
/// Scratch: 2 VGPRs (Out + Tmp), 2 SGPRs (s0 for NaN flag, s1 for exp-31).
uint32_t patchCvtF32Fp8(PatchContext &Ctx, size_t Idx) {
  const InternalDecodedInst &DI = Ctx.Decoded[Idx];
  // VOP1 (4 bytes) has no CLAMP bit; only VOP3 (8 bytes) needs patching.
  if (DI.Size != 8)
    return 0;

  const MCInst &Inst = DI.Inst;
  const VOP3Fp8UnpackLayout &L = UnpackFp8Layout;

  if (Inst.getNumOperands() < L.NumOperands) {
    log() << "hotswap: error: cvt_f32_fp8: operand count mismatch: "
          << "expected " << L.NumOperands << ", got " << Inst.getNumOperands()
          << " at offset 0x" << utohexstr(DI.Offset) << "\n";
    return 0;
  }

  if (!Inst.getOperand(L.Clamp).isImm() ||
      Inst.getOperand(L.Clamp).getImm() == 0)
    return 0;

  unsigned ByteSel = Inst.getOperand(L.ByteSel).isImm()
                         ? Inst.getOperand(L.ByteSel).getImm()
                         : 0;

  const MCRegisterInfo &MRI = *Ctx.LS.MRI;
  if (!Inst.getOperand(L.VDst).isReg() || !Inst.getOperand(L.Src0).isReg()) {
    log() << "hotswap: error: cvt_f32_fp8: unexpected imm operand at 0x"
          << utohexstr(DI.Offset) << "\n";
    return 0;
  }
  std::string VdstStr = toAsmRegName(MRI, Inst.getOperand(L.VDst).getReg());
  std::string Src0Str = toAsmRegName(MRI, Inst.getOperand(L.Src0).getReg());

  ScratchAllocation SA = allocateScratch(Ctx, Idx);

  // 2 scratch VGPRs: Out (byte extraction to F16 path to result),
  // Tmp (exp-31 direct construction + NaN constant).
  std::optional<unsigned> Out = SA.VgprAlloc.alloc();
  std::optional<unsigned> Tmp = SA.VgprAlloc.alloc();
  if (!Out || !Tmp) {
    log() << "hotswap: error: cvt_f32_fp8: unable to allocate 2 scratch "
          << "VGPRs at offset 0x" << utohexstr(DI.Offset) << "\n";
    return 0;
  }

  std::string OutName = vgprName(*Out);
  std::string TmpName = vgprName(*Tmp);

  std::optional<unsigned> NaNSgpr = SA.SgprAlloc.alloc();
  std::optional<unsigned> Exp31Sgpr = SA.SgprAlloc.alloc();
  std::optional<unsigned> VccSaveSgpr = SA.SgprAlloc.alloc();
  if (!NaNSgpr || !Exp31Sgpr || !VccSaveSgpr) {
    log() << "hotswap: error: cvt_f32_fp8: unable to allocate 3 scratch "
          << "SGPRs at offset 0x" << utohexstr(DI.Offset) << "\n";
    return 0;
  }

  std::string NaNName = sgprName(*NaNSgpr);
  std::string Exp31Name = sgprName(*Exp31Sgpr);
  std::string VccSaveName = sgprName(*VccSaveSgpr);

  std::string Asm;
  raw_string_ostream AsmOS(Asm);

  // Save VCC before clobbering it with v_cmp_* instructions.
  AsmOS << "s_mov_b32 " << VccSaveName << ", vcc_lo\n";

  // --- Byte extraction (byte_sel known at patch time) ---
  switch (ByteSel) {
  case 0:
    AsmOS << "v_and_b32 " << OutName << ", 0xFF, " << Src0Str << "\n";
    break;
  case 1:
    AsmOS << "v_bfe_u32 " << OutName << ", " << Src0Str << ", 8, 8\n";
    break;
  case 2:
    AsmOS << "v_bfe_u32 " << OutName << ", " << Src0Str << ", 16, 8\n";
    break;
  case 3:
    AsmOS << "v_lshrrev_b32 " << OutName << ", 24, " << Src0Str << "\n";
    break;
  }

  // --- NaN detection (byte == 0xFF) ---
  AsmOS << "v_cmp_eq_u32 0xFF, " << OutName << "\n";
  AsmOS << "s_mov_b32 " << NaNName << ", vcc_lo\n";

  // --- Exp-31 detection (byte >= 0xF8) ---
  AsmOS << "v_cmp_lt_u32 0xF7, " << OutName << "\n";
  AsmOS << "s_mov_b32 " << Exp31Name << ", vcc_lo\n";

  // --- Exp-31 direct F32 construction ---
  AsmOS << "v_and_b32 " << TmpName << ", 0x07, " << OutName << "\n";
  AsmOS << "v_lshlrev_b32 " << TmpName << ", 20, " << TmpName << "\n";
  AsmOS << "v_or_b32 " << TmpName << ", 0x47800000, " << TmpName << "\n";

  // --- F16 base path (handles bytes 0x00-0xF7 correctly) ---
  AsmOS << "v_lshlrev_b32 " << OutName << ", 7, " << OutName << "\n";
  AsmOS << "v_cvt_f32_f16 " << OutName << ", " << OutName << "\n";

  // --- Select exp-31 fixup ---
  AsmOS << "s_mov_b32 vcc_lo, " << Exp31Name << "\n";
  AsmOS << "v_cndmask_b32 " << OutName << ", " << OutName << ", " << TmpName
        << "\n";

  // --- NaN override (byte 0xFF to hardware qNaN 0x7FA3D000) ---
  AsmOS << "s_mov_b32 vcc_lo, " << NaNName << "\n";
  AsmOS << "v_mov_b32 " << TmpName << ", 0x7FA3D000\n";
  AsmOS << "v_cndmask_b32 " << VdstStr << ", " << OutName << ", " << TmpName
        << "\n";

  // Restore VCC to its pre-patch value.
  AsmOS << "s_mov_b32 vcc_lo, " << VccSaveName << "\n";

  SmallVector<uint8_t> ReplacementBytes = assembleInstructions(Asm, Ctx.LS);
  if (ReplacementBytes.empty()) {
    log() << "hotswap: error: cvt_f32_fp8: assembly failed for "
          << "replacement at offset 0x" << utohexstr(DI.Offset) << "\n";
    return 0;
  }

  unsigned ExtraV = SA.VgprAlloc.extraVgprsNeeded();
  if (checkKernelVgprBump(Ctx, SA.KernelName, ExtraV,
                          PatchRequirement::Optional) !=
      VgprBumpDecision::Apply)
    return 0;

  if (!emitToTrampoline(Ctx, DI.Offset, DI.Size, ReplacementBytes))
    return 0;

  if (!SA.KernelName.empty()) {
    KernelPatchStats &Stats = Ctx.KernelStats[SA.KernelName];
    Stats.ExtraVgprs = std::max(Stats.ExtraVgprs, ExtraV);
    Stats.ExtraSgprs =
        std::max(Stats.ExtraSgprs, SA.SgprAlloc.extraSgprsNeeded());
    Stats.ScratchReused += 2 - ExtraV;
    Stats.ScratchAboveKd += ExtraV;
  }

  ScratchPatchInfo Info;
  Info.Offset = DI.Offset;
  Info.ScratchRegs = SA.VgprAlloc.LiveAtPoint;
  Ctx.OutScratchPatches.push_back(std::move(Info));

  log() << "hotswap: cvt_f32_fp8: patched CLAMP=1 (E5M3) at offset 0x"
        << utohexstr(DI.Offset) << " (" << ReplacementBytes.size()
        << " bytes, scratch v" << *Out << "/v" << *Tmp
        << ", byte_sel=" << ByteSel << ")\n";

  return 1;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// applyScratchPatches - strong override
// ---------------------------------------------------------------------------

uint32_t applyScratchPatches(PatchContext &Ctx, size_t Idx) {
  StringRef Mnem(Ctx.Decoded[Idx].Mnemonic);

  if (Mnem == "v_cvt_pk_fp8_f32")
    return patchCvtPkFp8F32(Ctx, Idx);

  if (Mnem == "v_cvt_sr_fp8_f32")
    return patchCvtSrFp8F32(Ctx, Idx);

  // VOP1 mnemonic is "v_cvt_f32_fp8"; VOP3 may append "_e64" or other
  // suffixes depending on the LLVM build.  Use starts_with to match all
  // encoding variants; the Size and CLAMP checks inside patchCvtF32Fp8
  // filter out non-VOP3 and non-CLAMP forms.
  if (Mnem.starts_with("v_cvt_f32_fp8"))
    return patchCvtF32Fp8(Ctx, Idx);

  return 0;
}

void registerScratchPatch(HotswapPatchVTable &VT) {
  VT.applyScratchPatches = applyScratchPatches;
}

} // namespace hotswap
} // namespace COMGR
