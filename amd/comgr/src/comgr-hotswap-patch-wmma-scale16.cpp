//===- comgr-hotswap-patch-wmma-scale16.cpp - WMMA Scale16 decomposition --===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Scratch-patch pass for Case 3 of the B0-to-A0 scratch-patch pipeline:
/// WMMA Scale16 (block-16) to regular Scale (block-32) decomposition.
///
/// GFX1250 B0 uses VOP3PX3-encoded `v_wmma_scale16_f32_*` instructions with
/// 64-bit (B64) scale operands carrying block-16 scale granularity. A0 only
/// supports VOP3PX2-encoded `v_wmma_scale_f32_*` with 32-bit (B32) scale
/// operands at block-32 granularity. This file reduces block-16 scales to
/// block-32 via byte-pair max, then rewrites the encoding from VOP3PX3 to
/// VOP3PX2.
///
/// Covered instructions:
///   1. v_wmma_scale16_f32_16x16x128_f8f6f4 — decompose to regular Scale
///   2. v_wmma_scale16_f32_32x16x128_f4     — split 32x16 into 2× 16x16,
///      then decompose each half to regular Scale
///
/// Design document: docs/scratch-patches/3_wmma_scale16_decomp/README.md
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"

using namespace llvm;

namespace COMGR {
namespace hotswap {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::string vgprName(unsigned N) { return ("v" + Twine(N)).str(); }

// ---------------------------------------------------------------------------
// VOP3PX3 encoding field accessors
// ---------------------------------------------------------------------------
//
// VOP3PX3 and VOP3PX2 are both 128-bit (16-byte) fused instructions: an
// 8-byte LD_SCALE uop followed by an 8-byte base WMMA uop.  The encoding
// differences between VOP3PX3 (Scale16) and VOP3PX2 (regular Scale) are:
//   - Byte[2]: LD_SCALE opcode (0x3A for Scale16, 0x35 for regular Scale)
//   - SCALE_SRC field interpretation (B64 vs B32)
//
// Field positions within the LD_SCALE uop (first 8 bytes):
//   SCALE_SRC0: bits [40:32] = byte[4] bits [7:0] + byte[5] bit[0]
//   SCALE_SRC1: bits [49:41] = byte[5] bits [7:1] + byte[6] bits [1:0]
//
// Keep these as explicit byte masks rather than C++ bitfields: the fields
// cross byte boundaries, and bitfield layout is implementation-defined.

static unsigned extractScaleSrc0(const uint8_t *Raw) {
  return Raw[4] | ((Raw[5] & 0x01) << 8);
}

static unsigned extractScaleSrc1(const uint8_t *Raw) {
  return ((Raw[5] >> 1) & 0x7F) | ((Raw[6] & 0x03) << 7);
}

// Write a 9-bit SCALE_SRC0 value into bits [40:32], preserving all
// other bits.  Caller must ensure Raw points at 16+ writable bytes.
static void writeScaleSrc0(uint8_t *Raw, unsigned Enc) {
  Raw[4] = Enc & 0xFF;
  Raw[5] = (Raw[5] & 0xFE) | ((Enc >> 8) & 0x01);
}

// Write a 9-bit SCALE_SRC1 value into bits [49:41], preserving all
// other bits.  Must be called AFTER writeScaleSrc0 to avoid clobbering the
// shared byte [5].
static void writeScaleSrc1(uint8_t *Raw, unsigned Enc) {
  Raw[5] = (Raw[5] & 0x01) | ((Enc & 0x7F) << 1);
  Raw[6] = (Raw[6] & 0xFC) | ((Enc >> 7) & 0x03);
}

// AMDGPU SRC operand encoding: VGPRs are encoded as 256 + N.
static constexpr unsigned VgprEncBase = 256;

static bool isVgprEncoding(unsigned Enc) { return Enc >= VgprEncBase; }

static std::optional<unsigned> decodeVgprEncoding(unsigned Enc) {
  if (!isVgprEncoding(Enc))
    return std::nullopt;
  return Enc - VgprEncBase;
}

// ---------------------------------------------------------------------------
// Block-16 → block-32 scale reduction via VALU preamble
// ---------------------------------------------------------------------------
//
// Each B64 scale operand holds 8 × 8-bit block-16 scales across two VGPRs
// (Vn and Vn+1).  The reduction computes max(even, odd) for each adjacent
// byte pair, producing 4 × 8-bit block-32 scales in one VGPR:
//
//   max(Vn[7:0],     Vn[15:8])    → Vs[7:0]
//   max(Vn[23:16],   Vn[31:24])   → Vs[15:8]
//   max(Vn+1[7:0],   Vn+1[15:8])  → Vs[23:16]
//   max(Vn+1[23:16], Vn+1[31:24]) → Vs[31:24]
//
// Max-exponent is the recommended strategy for E8M0 scales (pure exponent,
// bias 127): taking the larger exponent ensures no element overflows the
// scale range.  For E5M3/E4M3 scales (only valid with FP4 matrices), max
// still provides a safe upper bound.

static void emitScaleReduction(raw_string_ostream &OS, StringRef SrcLo,
                               StringRef SrcHi, StringRef Dst, StringRef T1,
                               StringRef T2) {
  // Byte pair 0: max(SrcLo[7:0], SrcLo[15:8]) → Dst[7:0]
  OS << "v_and_b32 " << T1 << ", 0xFF, " << SrcLo << "\n";
  OS << "v_bfe_u32 " << T2 << ", " << SrcLo << ", 8, 8\n";
  OS << "v_max_u32 " << Dst << ", " << T1 << ", " << T2 << "\n";

  // Byte pair 1: max(SrcLo[23:16], SrcLo[31:24]) → Dst[15:8]
  OS << "v_bfe_u32 " << T1 << ", " << SrcLo << ", 16, 8\n";
  OS << "v_lshrrev_b32 " << T2 << ", 24, " << SrcLo << "\n";
  OS << "v_max_u32 " << T1 << ", " << T1 << ", " << T2 << "\n";
  OS << "v_lshl_or_b32 " << Dst << ", " << T1 << ", 8, " << Dst << "\n";

  // Byte pair 2: max(SrcHi[7:0], SrcHi[15:8]) → Dst[23:16]
  OS << "v_and_b32 " << T1 << ", 0xFF, " << SrcHi << "\n";
  OS << "v_bfe_u32 " << T2 << ", " << SrcHi << ", 8, 8\n";
  OS << "v_max_u32 " << T1 << ", " << T1 << ", " << T2 << "\n";
  OS << "v_lshl_or_b32 " << Dst << ", " << T1 << ", 16, " << Dst << "\n";

  // Byte pair 3: max(SrcHi[23:16], SrcHi[31:24]) → Dst[31:24]
  OS << "v_bfe_u32 " << T1 << ", " << SrcHi << ", 16, 8\n";
  OS << "v_lshrrev_b32 " << T2 << ", 24, " << SrcHi << "\n";
  OS << "v_max_u32 " << T1 << ", " << T1 << ", " << T2 << "\n";
  OS << "v_lshl_or_b32 " << Dst << ", " << T1 << ", 24, " << Dst << "\n";
}

// ---------------------------------------------------------------------------
// VOP3PX3 → VOP3PX2 encoding rewrite
// ---------------------------------------------------------------------------
//
// Both encodings are 128-bit (16-byte) fused instructions.  The rewrite:
//   1. Replaces byte[2] (LD_SCALE opcode: 0x3A → 0x35)
//   2. Replaces SCALE_SRC0/SCALE_SRC1 with scratch VGPR encodings
//
// The opcode constant is obtained by assembling a template VOP3PX2
// instruction, keeping the code free of hardcoded encoding bits.

static constexpr unsigned VOP3PXSize = 16;

static SmallVector<uint8_t> rewriteScale16ToScale(const uint8_t *OrigRaw,
                                                  unsigned OrigSize,
                                                  unsigned NewScaleSrc0Enc,
                                                  unsigned NewScaleSrc1Enc,
                                                  const LLVMState &LS) {
  SmallVector<uint8_t> Template = assembleSingleInst(
      "v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7], v[8:23], v[24:39], "
      "v[40:47], v48, v50",
      LS);
  if (Template.size() != VOP3PXSize) {
    log() << "hotswap: error: wmma_scale16: VOP3PX2 template assembly "
          << "produced " << Template.size() << " bytes (expected " << VOP3PXSize
          << ")\n";
    return {};
  }

  SmallVector<uint8_t> Rewritten(OrigRaw, OrigRaw + OrigSize);

  // The LD_SCALE opcode lives at byte[2]: 0x3A for Scale16 (VOP3PX3) and
  // 0x35 for regular Scale (VOP3PX2).  All other base WMMA encoding bytes
  // are identical between the two variants.
  Rewritten[2] = Template[2];

  writeScaleSrc0(Rewritten.data(), NewScaleSrc0Enc);

  // Must be called after writeScaleSrc0 because both share byte [5].
  writeScaleSrc1(Rewritten.data(), NewScaleSrc1Enc);

  // scale_src2 [58:50] = 0x100 (VGPR0). The field is architecturally
  // unused on VOP3PX2 but if left at 0 the SQ mis-decodes it as an SGPR
  // reference and stalls the SALU for 3 cycles. The in-place
  // applyVop3px2Src2Fix pass patches this on user-emitted v_wmma_scale_*
  // instructions in .text. Trampoline-emitted forms are NOT in Decoded[]
  // on the first rewrite pass, so the in-place fix doesn't see them; on
  // a second rewrite the trampolines have been appended to .text and the
  // fix fires, producing different bytes than pass 1 -- breaking
  // idempotency. Baking 0x100 here keeps wrap-emitted trampolines
  // bit-identical across passes and avoids the SALU stall. Same trick
  // PR #2 (VOP3PX2 wrap pass) uses in its LdScalePrefix bytes.
  Rewritten[6] &= 0x03; // clear scale_src2[5:0]
  Rewritten[7] =
      (Rewritten[7] & 0xF8) | 0x04; // set scale_src2[8]=1, clear [7:6]

  return Rewritten;
}

// ---------------------------------------------------------------------------
// v_wmma_scale16_f32_16x16x128_f8f6f4 → v_wmma_scale_f32_16x16x128_f8f6f4
// ---------------------------------------------------------------------------

static uint32_t patchWmmaScale16_16x16(PatchContext &Ctx, size_t Idx) {
  const InternalDecodedInst &DI = Ctx.Decoded[Idx];

  if (DI.Size != VOP3PXSize) {
    log() << "hotswap: error: wmma_scale16: unexpected inst size " << DI.Size
          << " at offset 0x" << utohexstr(DI.Offset) << "\n";
    return 0;
  }

  // Skip offsets another patch has already claimed (Trampoline entries
  // are appended to OutTrampolines before fixupTrampolineBranches
  // overwrites the original site with s_branch). Mirrors PR #2's wrap
  // pass; the previous `Decoded[Idx-1] == s_branch` heuristic never
  // fired meaningfully because Decoded[] is built from the original
  // .text and the dispatcher's mnemonic narrowing already filters out
  // sites the patch has rewritten on a re-rewrite.
  for (const Trampoline &T : Ctx.OutTrampolines)
    if (T.OriginalOffset == DI.Offset)
      return 0;

  const uint8_t *Raw = Ctx.Text + DI.Offset;

  unsigned ScaleSrc0Enc = extractScaleSrc0(Raw);
  unsigned ScaleSrc1Enc = extractScaleSrc1(Raw);

  std::optional<unsigned> Src0Base = decodeVgprEncoding(ScaleSrc0Enc);
  std::optional<unsigned> Src1Base = decodeVgprEncoding(ScaleSrc1Enc);
  if (!Src0Base || !Src1Base) {
    log() << "hotswap: error: wmma_scale16: unsupported non-VGPR scale "
          << "encoding at offset 0x" << utohexstr(DI.Offset) << "\n";
    return 0;
  }

  bool NeedReductionA = true;
  bool NeedReductionB = true;

  unsigned Src0Lo = *Src0Base;
  unsigned Src0Hi = Src0Lo + 1;
  unsigned Src1Lo = *Src1Base;
  unsigned Src1Hi = Src1Lo + 1;

  std::string KernelName =
      Ctx.Elf.findKernelAtAddress(DI.Offset + Ctx.Elf.textAddr());
  std::optional<unsigned> KdVgprs = Ctx.Elf.getKernelVgprCount(
      KernelName, getKernelVgprGranuleSize(Ctx, KernelName));
  unsigned KdCount = KdVgprs.value_or(Ctx.Config.MaxVgprs);

  VgprAllocator Alloc(Ctx.Liveness.LiveBefore[Idx], KdCount,
                      Ctx.Config.MaxVgprs);

  // Scratch allocation: 1 reduced-scale VGPR per operand needing reduction,
  // plus 2 shared temporaries for byte extraction/max.
  std::optional<unsigned> ScratchA, ScratchB, T1, T2;
  unsigned ScratchCount = 0;

  if (NeedReductionA) {
    ScratchA = Alloc.alloc();
    ++ScratchCount;
  }
  if (NeedReductionB) {
    ScratchB = Alloc.alloc();
    ++ScratchCount;
  }
  if (NeedReductionA || NeedReductionB) {
    T1 = Alloc.alloc();
    T2 = Alloc.alloc();
    ScratchCount += 2;
  }

  if ((NeedReductionA && !ScratchA) || (NeedReductionB && !ScratchB) ||
      ((NeedReductionA || NeedReductionB) && (!T1 || !T2))) {
    log() << "hotswap: error: wmma_scale16: unable to allocate " << ScratchCount
          << " scratch VGPRs at offset 0x" << utohexstr(DI.Offset) << "\n";
    return 0;
  }

  // --- Build VALU preamble for scale reduction ---
  std::string Asm;
  raw_string_ostream AsmOS(Asm);

  if (NeedReductionA)
    emitScaleReduction(AsmOS, vgprName(Src0Lo), vgprName(Src0Hi),
                       vgprName(*ScratchA), vgprName(*T1), vgprName(*T2));
  if (NeedReductionB)
    emitScaleReduction(AsmOS, vgprName(Src1Lo), vgprName(Src1Hi),
                       vgprName(*ScratchB), vgprName(*T1), vgprName(*T2));

  SmallVector<uint8_t> PreambleBytes;
  if (NeedReductionA || NeedReductionB) {
    PreambleBytes = assembleSingleInst(Asm, Ctx.LS);
    if (PreambleBytes.empty()) {
      log() << "hotswap: error: wmma_scale16: preamble assembly failed at "
            << "offset 0x" << utohexstr(DI.Offset) << "\n";
      return 0;
    }
  }

  // --- Rewrite the WMMA encoding from VOP3PX3 → VOP3PX2 ---
  unsigned NewSrc0Enc =
      NeedReductionA ? (VgprEncBase + *ScratchA) : ScaleSrc0Enc;
  unsigned NewSrc1Enc =
      NeedReductionB ? (VgprEncBase + *ScratchB) : ScaleSrc1Enc;

  SmallVector<uint8_t> WmmaBytes =
      rewriteScale16ToScale(Raw, DI.Size, NewSrc0Enc, NewSrc1Enc, Ctx.LS);
  if (WmmaBytes.empty())
    return 0;

  // --- Concatenate: VALU preamble + rewritten WMMA → replacement ---
  SmallVector<uint8_t> Replacement;
  Replacement.insert(Replacement.end(), PreambleBytes.begin(),
                     PreambleBytes.end());
  Replacement.insert(Replacement.end(), WmmaBytes.begin(), WmmaBytes.end());

  unsigned Extra = Alloc.extraVgprsNeeded();
  if (checkKernelVgprBump(Ctx, KernelName, Extra, PatchRequirement::Optional) !=
      VgprBumpDecision::Apply)
    return 0;

  if (!emitToTrampoline(Ctx, DI.Offset, DI.Size, Replacement))
    return 0;

  KernelPatchStats &Stats = Ctx.KernelStats[KernelName];
  if (Extra > Stats.ExtraVgprs)
    Stats.ExtraVgprs = Extra;
  Stats.ScratchReused += ScratchCount - Extra;
  Stats.ScratchAboveKd += Extra;

  ScratchPatchInfo Info;
  Info.Offset = DI.Offset;
  Info.ScratchRegs = Alloc.LiveAtPoint;
  Ctx.OutScratchPatches.push_back(std::move(Info));

  log() << "hotswap: wmma_scale16: patched 16x16 Scale16→Scale at offset 0x"
        << utohexstr(DI.Offset) << " (" << Replacement.size()
        << " bytes, reductionA=" << NeedReductionA
        << ", reductionB=" << NeedReductionB << ")\n";

  return 1;
}

// ---------------------------------------------------------------------------
// Base WMMA uop register field extraction (bytes 8-15 of 16-byte VOP3PX)
// ---------------------------------------------------------------------------
//
// The base WMMA uop follows standard VOP3P field layout:
//   VDST:  byte[8] bits [7:0] (8-bit, raw VGPR number without +256)
//   SRC0:  byte[12] bits [7:0] + byte[13] bit[0] << 8 (9-bit)
//   SRC1:  byte[13] bits [7:1] >> 1 + byte[14] bits [1:0] << 7 (9-bit)

static unsigned extractVdst(const uint8_t *Raw) { return Raw[8]; }

static unsigned extractSrc0(const uint8_t *Raw) {
  return Raw[12] | ((Raw[13] & 0x01u) << 8);
}

static unsigned extractSrc1(const uint8_t *Raw) {
  return ((Raw[13] >> 1) & 0x7Fu) | ((Raw[14] & 0x03u) << 7);
}

// SRC2 (9-bit): bits [122:114] of the 128-bit VOP3PX instruction
// (per VOP3PInstructions.td let Inst{122-114} = src2).
//   src2[5:0] = byte[14] bits [7:2]
//   src2[8:6] = byte[15] bits [2:0]
static unsigned extractSrc2(const uint8_t *Raw) {
  return ((Raw[14] >> 2) & 0x3Fu) | ((Raw[15] & 0x07u) << 6);
}

// ---------------------------------------------------------------------------
// VOP3PX3 modifier field extraction
// ---------------------------------------------------------------------------
//
// SCALE_OPSEL_HI[0] (matrix_b_scale, thread-half for B): bit 59 = byte[7] bit 3
// matrix_a_scale_fmt[1:0]: bits [62:61] = byte[7] bits [6:5]
// matrix_b_scale_fmt[1:0]: bits [9:8] = byte[1] bits [1:0]

static bool extractBScaleRow1(const uint8_t *Raw) { return (Raw[7] >> 3) & 1; }

static unsigned extractAScaleFmt(const uint8_t *Raw) {
  return (Raw[7] >> 5) & 0x3;
}

static unsigned extractBScaleFmt(const uint8_t *Raw) { return Raw[1] & 0x3; }

// VOP3PX neg_lo / neg_hi bits live in the WMMA uop's modifier fields
// (bytes 8..15), per VOP3PInstructions.td VOP3PX2e:
//   Inst{72} = neg_hi src0   -> byte[9]  bit 0
//   Inst{73} = neg_hi src1   -> byte[9]  bit 1
//   Inst{74} = neg_hi src2   -> byte[9]  bit 2
//   Inst{125} = neg_lo src0  -> byte[15] bit 5
//   Inst{126} = neg_lo src1  -> byte[15] bit 6
//   Inst{127} = neg_lo src2  -> byte[15] bit 7
//
// For the wmma_scale16_f32_32x16x128_f4 source profile only src2's neg_lo
// is wired by the intrinsic (it's the c_mod=NEG path: builtin's c_mod
// short maps to src2_modifiers and the assembler prints `neg_lo:[0,0,1]`
// when the bit is set). src0/src1 neg bits are extracted defensively in
// case a future profile wires them.
//
// Returns a 6-bit packed value with bit layout matching the AMDGPU printer
// list order [src0, src1, src2] for each of {neg_lo, neg_hi}:
//   bit 0 = neg_lo src0, bit 1 = neg_lo src1, bit 2 = neg_lo src2,
//   bit 3 = neg_hi src0, bit 4 = neg_hi src1, bit 5 = neg_hi src2.
static unsigned extractNegFlags(const uint8_t *Raw) {
  unsigned NegLo = (Raw[15] >> 5) & 0x7; // src0/1/2 neg_lo
  unsigned NegHi = Raw[9] & 0x7;         // src0/1/2 neg_hi
  return NegLo | (NegHi << 3);
}

// Format a scale operand for assembly output. Returns the operand string
// (e.g. "v42" for VGPRs, "s0" for SGPRs). Returns "" on unsupported encoding.
static std::string formatScaleOp(unsigned Enc,
                                 std::optional<unsigned> ReducedVgpr) {
  if (ReducedVgpr)
    return vgprName(*ReducedVgpr);
  if (Enc >= VgprEncBase)
    return vgprName(Enc - VgprEncBase);
  if (Enc <= 105)
    return ("s" + Twine(Enc)).str();
  return "";
}

// Format the original SRC2/C operand for one M-split half.
//
// The source 32x16 form's C operand may be:
//   - An inline integer immediate (encodings 128..208 = ints -16..64;
//     0 = encoding 128 is the common case when the kernel uses an
//     all-zero accumulator that the compiler folds to inline-imm 0).
//   - An inline float immediate (encodings 240..247 cover 0.5, -0.5,
//     1.0, -1.0, 2.0, -2.0, 4.0, -4.0).
//   - A VGPR range v[c:c+15] (encoding >= 256).
//   - An SGPR (encoding 0..105; not expected for WMMA accumulators).
//
// For inline immediates the same value applies to both halves (each half
// has its own accumulator slice on M-split, no carry between halves).
// For a VGPR range the source's v[c:c+15] is sliced to v[c:c+7] for
// half-0 and v[c+8:c+15] for half-1, mirroring how dst is split.
//
// Returns "" if the encoding falls in a range we don't know how to print
// (the caller logs an error and bails). Inline integer / float immediates
// share the standard AMDGPU encoding numbering -- the assembler accepts
// either the decoded literal (e.g. "0", "1.0") or the raw encoding
// printed as decimal.
static std::string formatSrc2(unsigned Enc, unsigned Half) {
  if (Enc >= VgprEncBase) {
    unsigned Base = (Enc - VgprEncBase) + Half * 8;
    return ("v[" + Twine(Base) + ":" + Twine(Base + 7) + "]").str();
  }
  // Integer inline immediates: 128 + N for N in [-16, 64].
  if (Enc >= 128 && Enc <= 192)
    return Twine(static_cast<int>(Enc) - 128).str();
  if (Enc >= 193 && Enc <= 208)
    return Twine(192 - static_cast<int>(Enc)).str();
  // Float inline immediates: 240=0.5, 241=-0.5, 242=1.0, 243=-1.0,
  // 244=2.0, 245=-2.0, 246=4.0, 247=-4.0.
  switch (Enc) {
  case 240:
    return "0.5";
  case 241:
    return "-0.5";
  case 242:
    return "1.0";
  case 243:
    return "-1.0";
  case 244:
    return "2.0";
  case 245:
    return "-2.0";
  case 246:
    return "4.0";
  case 247:
    return "-4.0";
  }
  // SGPRs and unrecognized encodings fall through. WMMA accumulators are
  // architecturally VGPR-or-immediate; an SGPR here would be malformed input.
  return "";
}

// ---------------------------------------------------------------------------
// v_wmma_scale16_f32_32x16x128_f4 → 2× v_wmma_scale_f32_16x16x128_f8f6f4
// ---------------------------------------------------------------------------
//
// The 32x16 FP4 instruction is B0-only. Decompose it into two 16x16 halves
// that split along the M dimension (rows), then each half undergoes the same
// block-16→block-32 scale reduction as the 16x16 case.
//
// Register mapping (linear VGPR packing):
//   Half 0: D=v[d:d+7],   A=v[a:a+7],   B=v[b:b+7], C=<src2-half-0>
//   Half 1: D=v[d+8:d+15], A=v[a+8:a+15], B=v[b:b+7], C=<src2-half-1>
//
// C tracks the source's src2 operand:
//   - inline-imm (e.g. 0 when the kernel accumulator is folded by clang):
//     same literal on both halves (no accumulator carry on M-split)
//   - VGPR range v[c:c+15]: sliced to v[c:c+7] / v[c+8:c+15]
// See formatSrc2 above.
//
// Scale operands are shared: both halves use the same reduced B32 scale.
// SCALE_OPSEL[0] (matrix_a_scale) selects the thread-half for A scales:
//   Half 0 → ROW0 (threads 0-15), Half 1 → ROW1 (threads 16-31).

static uint32_t patchWmmaScale16_32x16(PatchContext &Ctx, size_t Idx) {
  const InternalDecodedInst &DI = Ctx.Decoded[Idx];

  if (DI.Size != VOP3PXSize) {
    log() << "hotswap: error: wmma_scale16: unexpected 32x16 inst size "
          << DI.Size << " at offset 0x" << utohexstr(DI.Offset) << "\n";
    return 0;
  }

  // Skip offsets another patch has already claimed. Mirrors the 16x16
  // path (see patchWmmaScale16_16x16 for the rationale).
  for (const Trampoline &T : Ctx.OutTrampolines)
    if (T.OriginalOffset == DI.Offset)
      return 0;

  const uint8_t *Raw = Ctx.Text + DI.Offset;

  unsigned DBase = extractVdst(Raw);
  unsigned Src0Enc = extractSrc0(Raw);
  unsigned Src1Enc = extractSrc1(Raw);

  if (!isVgprEncoding(Src0Enc) || !isVgprEncoding(Src1Enc)) {
    log() << "hotswap: error: wmma_scale16: 32x16 non-VGPR matrix src at "
          << "offset 0x" << utohexstr(DI.Offset) << "\n";
    return 0;
  }

  unsigned ABase = Src0Enc - VgprEncBase;
  unsigned BBase = Src1Enc - VgprEncBase;
  unsigned Src2Enc = extractSrc2(Raw);

  unsigned ScaleSrc0Enc = extractScaleSrc0(Raw);
  unsigned ScaleSrc1Enc = extractScaleSrc1(Raw);

  std::optional<unsigned> ScaleSrc0Base = decodeVgprEncoding(ScaleSrc0Enc);
  std::optional<unsigned> ScaleSrc1Base = decodeVgprEncoding(ScaleSrc1Enc);
  if (!ScaleSrc0Base || !ScaleSrc1Base) {
    log() << "hotswap: error: wmma_scale16: 32x16 unsupported non-VGPR "
          << "scale encoding at offset 0x" << utohexstr(DI.Offset) << "\n";
    return 0;
  }

  bool NeedReductionA = true;
  bool NeedReductionB = true;

  unsigned Src0Lo = *ScaleSrc0Base;
  unsigned Src0Hi = Src0Lo + 1;
  unsigned Src1Lo = *ScaleSrc1Base;
  unsigned Src1Hi = Src1Lo + 1;

  bool OrigBScaleRow1 = extractBScaleRow1(Raw);
  unsigned AScaleFmt = extractAScaleFmt(Raw);
  unsigned BScaleFmt = extractBScaleFmt(Raw);
  unsigned NegFlags = extractNegFlags(Raw);

  std::string KernelName =
      Ctx.Elf.findKernelAtAddress(DI.Offset + Ctx.Elf.textAddr());
  std::optional<unsigned> KdVgprs = Ctx.Elf.getKernelVgprCount(
      KernelName, getKernelVgprGranuleSize(Ctx, KernelName));
  unsigned KdCount = KdVgprs.value_or(Ctx.Config.MaxVgprs);

  VgprAllocator Alloc(Ctx.Liveness.LiveBefore[Idx], KdCount,
                      Ctx.Config.MaxVgprs);

  std::optional<unsigned> ScratchA, ScratchB, T1, T2;
  unsigned ScratchCount = 0;

  if (NeedReductionA) {
    ScratchA = Alloc.alloc();
    ++ScratchCount;
  }
  if (NeedReductionB) {
    ScratchB = Alloc.alloc();
    ++ScratchCount;
  }
  if (NeedReductionA || NeedReductionB) {
    T1 = Alloc.alloc();
    T2 = Alloc.alloc();
    ScratchCount += 2;
  }

  if ((NeedReductionA && !ScratchA) || (NeedReductionB && !ScratchB) ||
      ((NeedReductionA || NeedReductionB) && (!T1 || !T2))) {
    log() << "hotswap: error: wmma_scale16: unable to allocate " << ScratchCount
          << " scratch VGPRs for 32x16 at offset 0x" << utohexstr(DI.Offset)
          << "\n";
    return 0;
  }

  // --- Scale reduction preamble (shared by both halves) ---
  std::string Asm;
  raw_string_ostream AsmOS(Asm);

  if (NeedReductionA)
    emitScaleReduction(AsmOS, vgprName(Src0Lo), vgprName(Src0Hi),
                       vgprName(*ScratchA), vgprName(*T1), vgprName(*T2));
  if (NeedReductionB)
    emitScaleReduction(AsmOS, vgprName(Src1Lo), vgprName(Src1Hi),
                       vgprName(*ScratchB), vgprName(*T1), vgprName(*T2));

  SmallVector<uint8_t> PreambleBytes;
  if (NeedReductionA || NeedReductionB) {
    PreambleBytes = assembleSingleInst(Asm, Ctx.LS);
    if (PreambleBytes.empty()) {
      log() << "hotswap: error: wmma_scale16: 32x16 preamble assembly failed "
            << "at offset 0x" << utohexstr(DI.Offset) << "\n";
      return 0;
    }
  }

  // --- Assemble two 16x16 WMMA halves ---
  std::string ScaleAStr = formatScaleOp(ScaleSrc0Enc, ScratchA);
  std::string ScaleBStr = formatScaleOp(ScaleSrc1Enc, ScratchB);

  if (ScaleAStr.empty() || ScaleBStr.empty()) {
    log() << "hotswap: error: wmma_scale16: 32x16 unsupported scale encoding "
          << "at offset 0x" << utohexstr(DI.Offset) << "\n";
    return 0;
  }

  SmallVector<uint8_t> Replacement;
  Replacement.insert(Replacement.end(), PreambleBytes.begin(),
                     PreambleBytes.end());

  for (unsigned Half = 0; Half < 2; ++Half) {
    unsigned HalfD = DBase + Half * 8;
    unsigned HalfA = ABase + Half * 8;

    // C (src2) tracks the source's encoding: for an inline immediate the
    // same literal applies to both halves (M-split has no accumulator
    // carry between halves); for a VGPR range we slice it the same way
    // as D. Always sourcing C from HalfD is wrong when the kernel
    // accumulator is folded to inline-imm 0 -- the WMMA would then read
    // arbitrary stale bytes from D's range as the accumulator input.
    std::string CStr = formatSrc2(Src2Enc, Half);
    if (CStr.empty()) {
      log() << "hotswap: error: wmma_scale16: 32x16 unsupported src2 "
            << "encoding 0x" << utohexstr(Src2Enc) << " at offset 0x"
            << utohexstr(DI.Offset) << "\n";
      return 0;
    }

    std::string WmmaAsm;
    raw_string_ostream WOS(WmmaAsm);

    WOS << "v_wmma_scale_f32_16x16x128_f8f6f4" << " v[" << HalfD << ":"
        << (HalfD + 7) << "]," << " v[" << HalfA << ":" << (HalfA + 7) << "],"
        << " v[" << BBase << ":" << (BBase + 7) << "]," << " " << CStr << ","
        << " " << ScaleAStr << ", " << ScaleBStr
        << " matrix_a_fmt:MATRIX_FMT_FP4 matrix_b_fmt:MATRIX_FMT_FP4";

    if (Half == 1)
      WOS << " matrix_a_scale:MATRIX_SCALE_ROW1";
    if (OrigBScaleRow1)
      WOS << " matrix_b_scale:MATRIX_SCALE_ROW1";
    if (AScaleFmt == 1)
      WOS << " matrix_a_scale_fmt:MATRIX_SCALE_FMT_E5M3";
    else if (AScaleFmt == 2)
      WOS << " matrix_a_scale_fmt:MATRIX_SCALE_FMT_E4M3";
    if (BScaleFmt == 1)
      WOS << " matrix_b_scale_fmt:MATRIX_SCALE_FMT_E5M3";
    else if (BScaleFmt == 2)
      WOS << " matrix_b_scale_fmt:MATRIX_SCALE_FMT_E4M3";

    // Propagate per-source neg_lo / neg_hi modifiers to both halves.
    // M-split has no accumulator carry between halves, so each half
    // gets the full set of source modifiers verbatim. The printer
    // formats these as `neg_lo:[a,b,c]` / `neg_hi:[a,b,c]` lists; we
    // emit them only when at least one bit in the corresponding triple
    // is set, since `neg_lo:[0,0,0]` is the default and most asm
    // dialects omit it.
    unsigned NegLo = NegFlags & 0x7;
    unsigned NegHi = (NegFlags >> 3) & 0x7;
    if (NegLo != 0)
      WOS << " neg_lo:[" << ((NegLo >> 0) & 1) << "," << ((NegLo >> 1) & 1)
          << "," << ((NegLo >> 2) & 1) << "]";
    if (NegHi != 0)
      WOS << " neg_hi:[" << ((NegHi >> 0) & 1) << "," << ((NegHi >> 1) & 1)
          << "," << ((NegHi >> 2) & 1) << "]";

    WOS << "\n";

    SmallVector<uint8_t> HalfBytes = assembleSingleInst(WmmaAsm, Ctx.LS);
    if (HalfBytes.size() != VOP3PXSize) {
      log() << "hotswap: error: wmma_scale16: 32x16 half " << Half
            << " assembly produced " << HalfBytes.size() << " bytes (expected "
            << VOP3PXSize << ") at offset 0x" << utohexstr(DI.Offset) << "\n";
      return 0;
    }

    // Same scale_src2 = 0x100 (VGPR0) bake-in as the 16x16 path. The
    // assembler leaves scale_src2 = 0x80 (SGPR mis-decoded) by default;
    // applyVop3px2Src2Fix would patch it on a second rewrite (breaking
    // idempotency) and the SALU would stall 3 cycles per invocation on
    // first execution. See rewriteScale16ToScale's comment for the full
    // story.
    HalfBytes[6] &= 0x03;
    HalfBytes[7] = (HalfBytes[7] & 0xF8) | 0x04;

    Replacement.insert(Replacement.end(), HalfBytes.begin(), HalfBytes.end());
  }

  unsigned Extra = Alloc.extraVgprsNeeded();
  if (checkKernelVgprBump(Ctx, KernelName, Extra, PatchRequirement::Optional) !=
      VgprBumpDecision::Apply)
    return 0;

  if (!emitToTrampoline(Ctx, DI.Offset, DI.Size, Replacement))
    return 0;

  KernelPatchStats &Stats = Ctx.KernelStats[KernelName];
  if (Extra > Stats.ExtraVgprs)
    Stats.ExtraVgprs = Extra;
  Stats.ScratchReused += ScratchCount - Extra;
  Stats.ScratchAboveKd += Extra;

  ScratchPatchInfo Info;
  Info.Offset = DI.Offset;
  Info.ScratchRegs = Alloc.LiveAtPoint;
  Ctx.OutScratchPatches.push_back(std::move(Info));

  log() << "hotswap: wmma_scale16: patched 32x16→2x16x16 Scale16→Scale at "
        << "offset 0x" << utohexstr(DI.Offset) << " (" << Replacement.size()
        << " bytes, reductionA=" << NeedReductionA
        << ", reductionB=" << NeedReductionB << ")\n";

  return 1;
}

// ---------------------------------------------------------------------------
// patchWmmaScale16 — dispatch for WMMA Scale16 variants
// ---------------------------------------------------------------------------

static uint32_t patchWmmaScale16(PatchContext &Ctx, size_t Idx) {
  StringRef Mnem(Ctx.Decoded[Idx].Mnemonic);

  if (Mnem == "v_wmma_scale16_f32_16x16x128_f8f6f4")
    return patchWmmaScale16_16x16(Ctx, Idx);

  if (Mnem == "v_wmma_scale16_f32_32x16x128_f4")
    return patchWmmaScale16_32x16(Ctx, Idx);

  return 0;
}

// ---------------------------------------------------------------------------
// applyWmmaScale16Patches
// ---------------------------------------------------------------------------
//
// Called once per decoded instruction during the rewrite loop. Returns the
// number of patches applied (0 or 1).

static uint32_t applyWmmaScale16PatchesImpl(PatchContext &Ctx, size_t Idx) {
  StringRef Mnem(Ctx.Decoded[Idx].Mnemonic);
  if (Mnem.starts_with("v_wmma_scale16_f32_"))
    return patchWmmaScale16(Ctx, Idx);
  return 0;
}

void registerWmmaScale16Patch(HotswapPatchVTable &VT) {
  VT.applyWmmaScale16Patches = &applyWmmaScale16PatchesImpl;
}

} // namespace hotswap
} // namespace COMGR
