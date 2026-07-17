//===- comgr-hotswap-patch-wmma-scale16.cpp - WMMA Scale16 decomposition --===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Lowers block-16 scaled WMMA (v_wmma_scale16_f32_*) for gfx1250 hardware that
/// only has block-32 scaled WMMA (v_wmma_scale_f32_*). Done exactly, or failing
/// closed when it cannot be applied.
///
/// A block-32 op applies one (scaleA, scaleB) pair across all 32 K-elements of
/// a block, so it cannot honor both block-16 sub-scales of that block at once.
/// The earlier approach collapsed each sub-scale pair with a byte-pair max,
/// which scaled the smaller half by a power of two and silently miscompiled
/// scaled kernels.
///
/// Exact lowering (K-split): the scale is applied per block after the dot and
/// before the accumulate, so we split each block-16 WMMA into two block-32
/// WMMAs chained through the accumulator, each seeing one 16-wide K-subblock:
///
///   pass-low : A' = low-16 K-subblock of A, rest zeroed; even scale bytes;
///              write D (src2 = original C).
///   pass-high: A' = high-16 K-subblock of A, rest zeroed; odd scale bytes;
///              accumulate (src2 = D).
///
/// Masking A alone suffices since A==0 => A*B==0. How a 16-K subblock maps to
/// lanes or VGPRs depends on the matrix-A format:
///   * FP8/BF8: subblocks split by wave lane, so a lane mask isolates one.
///   * FP4/FP6/BF6: a whole 32-block sits in one lane group and the split runs
///     along the VGPR index, so we null the opposite subblock's VGPRs (a lane
///     mask would wrongly zero whole 32-blocks).
/// Each pass's block-32 scale is a byte-gather of the block-16 scale bytes:
/// even bytes feed the low subblocks, odd bytes the high ones.
///
/// The masked A copy lands in a contiguous VGPR block allocated above the
/// kernel's VGPR count and below MaxVgprs (256 on GFX1250), so it stays in VGPR
/// bank 0 and needs no s_set_vgpr_msb switch.
///
/// Fail-closed fallback: when the scratch budget (A-width VGPRs plus a few
/// scale/temp VGPRs and one scratch SGPR) is unavailable, the pass marks the
/// patch failed so the rewrite returns an error instead of a miscompile. A loud
/// failure beats silent wrong results.
///
/// The 32x16x128_f4 (M=32) variant also needs an M-split; it is not lowered
/// exactly yet and fails closed.
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace COMGR {
namespace hotswap {

// Both Scale16 (VOP3PX3) and regular Scale (VOP3PX2) are 128-bit (16-byte)
// fused instructions: an 8-byte LD_SCALE uop followed by an 8-byte base WMMA
// uop.
static constexpr unsigned VOP3PXSize = 16;

// AMDGPU SRC operand encoding: VGPRs are 256 + N.
static constexpr unsigned VgprEncBase = 256;

static std::string vgprName(unsigned N) { return ("v" + Twine(N)).str(); }

static bool isVgprEncoding(unsigned Enc) { return Enc >= VgprEncBase; }

static std::optional<unsigned> decodeVgprEncoding(unsigned Enc) {
  if (!isVgprEncoding(Enc))
    return std::nullopt;
  return Enc - VgprEncBase;
}

// -- LD_SCALE uop field accessors (bytes 0-7) --------------------------------
//   SCALE_SRC0: bits [40:32] = byte[4] + byte[5] bit[0]
//   SCALE_SRC1: bits [49:41] = byte[5] bits[7:1] + byte[6] bits[1:0]

static unsigned extractScaleSrc0(const uint8_t *Raw) {
  return Raw[4] | ((Raw[5] & 0x01) << 8);
}

static unsigned extractScaleSrc1(const uint8_t *Raw) {
  return ((Raw[5] >> 1) & 0x7F) | ((Raw[6] & 0x03) << 7);
}

static void writeScaleSrc0(uint8_t *Raw, unsigned Enc) {
  Raw[4] = Enc & 0xFF;
  Raw[5] = (Raw[5] & 0xFE) | ((Enc >> 8) & 0x01);
}

// Must be called after writeScaleSrc0 (both share byte[5]).
static void writeScaleSrc1(uint8_t *Raw, unsigned Enc) {
  Raw[5] = (Raw[5] & 0x01) | ((Enc & 0x7F) << 1);
  Raw[6] = (Raw[6] & 0xFC) | ((Enc >> 7) & 0x03);
}

// -- Base WMMA uop field accessors (bytes 8-15) ------------------------------
//   VDST: byte[8] (8-bit raw VGPR number, no +256)
//   SRC0: byte[12] + byte[13] bit[0] (9-bit; matrix A)
//   SRC2: byte[14] bits[7:2] + byte[15] bits[2:0] (9-bit; accumulator C)

static unsigned extractVdst(const uint8_t *Raw) { return Raw[8]; }

static void writeSrc0(uint8_t *Raw, unsigned Enc) {
  Raw[12] = Enc & 0xFF;
  Raw[13] = (Raw[13] & 0xFE) | ((Enc >> 8) & 0x01);
}

static void writeSrc2(uint8_t *Raw, unsigned Enc) {
  Raw[14] = (Raw[14] & 0x03) | ((Enc & 0x3F) << 2);
  Raw[15] = (Raw[15] & 0xF8) | ((Enc >> 6) & 0x07);
}

// -- VOP3PX3 -> VOP3PX2 encoding rewrite -------------------------------------
//
// Turns a block-16 (VOP3PX3) scaled WMMA into a block-32 (VOP3PX2) one: copies
// the 16-byte instruction, swaps the LD_SCALE opcode byte (taken from a
// template assembly so no opcode bits are hardcoded), writes the new block-32
// scale sources, and bakes scale_src2 = VGPR0. scale_src2 is unused on
// VOP3PX2, but leaving it 0 makes the SQ mis-decode it as an SGPR and stall;
// baking it also keeps the bytes idempotent across passes. All other base-WMMA
// bytes (VDST, SRC0/1/2, matrix formats, neg modifiers) survive the byte copy
// and are patched by the caller.
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
  Rewritten[2] = Template[2];
  writeScaleSrc0(Rewritten.data(), NewScaleSrc0Enc);
  writeScaleSrc1(Rewritten.data(), NewScaleSrc1Enc);
  Rewritten[6] &= 0x03;                        // clear scale_src2[5:0]
  Rewritten[7] = (Rewritten[7] & 0xF8) | 0x04; // scale_src2[8]=1, clear [7:6]
  return Rewritten;
}

// -- Block-16 scale byte-gather (deinterleave) -------------------------------
//
// Each B64 scale operand holds 8 8-bit block-16 scales across Vn (bytes 0-3)
// and Vn+1 (bytes 4-7). The block-32 scale for K-block j (j=0..3) is the
// low-subblock scale (even byte 2j) for pass-low and the high-subblock scale
// (odd byte 2j+1) for pass-high, packed into one VGPR as
// [byte0..3] = k-block 0..3.

static void emitGatherEven(raw_string_ostream &OS, StringRef Lo, StringRef Hi,
                           StringRef Dst, StringRef T) {
  // Dst = { Lo[7:0], Lo[23:16], Hi[7:0], Hi[23:16] } (bytes 0,2,4,6)
  OS << "v_and_b32 " << Dst << ", 0xff, " << Lo << "\n";
  OS << "v_bfe_u32 " << T << ", " << Lo << ", 16, 8\n";
  OS << "v_lshl_or_b32 " << Dst << ", " << T << ", 8, " << Dst << "\n";
  OS << "v_and_b32 " << T << ", 0xff, " << Hi << "\n";
  OS << "v_lshl_or_b32 " << Dst << ", " << T << ", 16, " << Dst << "\n";
  OS << "v_bfe_u32 " << T << ", " << Hi << ", 16, 8\n";
  OS << "v_lshl_or_b32 " << Dst << ", " << T << ", 24, " << Dst << "\n";
}

static void emitGatherOdd(raw_string_ostream &OS, StringRef Lo, StringRef Hi,
                          StringRef Dst, StringRef T) {
  // Dst = { Lo[15:8], Lo[31:24], Hi[15:8], Hi[31:24] } (bytes 1,3,5,7)
  OS << "v_bfe_u32 " << Dst << ", " << Lo << ", 8, 8\n";
  OS << "v_bfe_u32 " << T << ", " << Lo << ", 24, 8\n";
  OS << "v_lshl_or_b32 " << Dst << ", " << T << ", 8, " << Dst << "\n";
  OS << "v_bfe_u32 " << T << ", " << Hi << ", 8, 8\n";
  OS << "v_lshl_or_b32 " << Dst << ", " << T << ", 16, " << Dst << "\n";
  OS << "v_lshrrev_b32 " << T << ", 24, " << Hi << "\n";
  OS << "v_lshl_or_b32 " << Dst << ", " << T << ", 24, " << Dst << "\n";
}

// A' = mask ? A : 0, per lane, for W consecutive VGPRs from ABase into SBase.
// MaskImm selects the wave lanes to keep (0x0000FFFF = lanes 0-15).
//
// FP8/BF8 only: a K=32 block's low-16 K-subblock lives in lanes 0-15 and the
// high-16 in lanes 16-31, so a lane mask isolates a subblock.
static void emitLaneMaskCopy(raw_string_ostream &OS, StringRef MaskSgpr,
                             uint32_t MaskImm, unsigned SBase, unsigned ABase,
                             unsigned W) {
  OS << "s_mov_b32 " << MaskSgpr << ", 0x" << utohexstr(MaskImm) << "\n";
  for (unsigned I = 0; I < W; ++I)
    OS << "v_cndmask_b32_e64 " << vgprName(SBase + I) << ", 0, "
       << vgprName(ABase + I) << ", " << MaskSgpr << "\n";
}

// A' keeps the VGPRs of the low (KeepLow=true) or high 16-K subblocks and zeros
// the rest, copying W consecutive VGPRs from ABase into SBase.
//
// FP4/FP6/BF6: a whole K=32 block sits in one lane group and the low-16/high-16
// split runs along the VGPR index. Subblocks are SubW consecutive VGPRs (FP4=2,
// FP6=3); even-indexed ones are the low halves, odd-indexed the high. A lane
// mask would wrongly zero whole 32-blocks here, so we null the opposite
// subblock's VGPRs instead.
static void emitVgprSelectCopy(raw_string_ostream &OS, bool KeepLow,
                               unsigned SBase, unsigned ABase, unsigned W,
                               unsigned SubW) {
  for (unsigned I = 0; I < W; ++I) {
    bool IsLow = ((I / SubW) % 2) == 0;
    if (IsLow == KeepLow)
      OS << "v_mov_b32 " << vgprName(SBase + I) << ", " << vgprName(ABase + I)
         << "\n";
    else
      OS << "v_mov_b32 " << vgprName(SBase + I) << ", 0\n";
  }
}

// Parse the matrix-A (src0) VGPR range from the printer's canonical form.
struct VgprRange {
  unsigned Base;
  unsigned Width;
};

static std::optional<VgprRange>
matrixAOperandRange(PatchContext &Ctx, const InternalDecodedInst &DI) {
  SmallString<256> Buf;
  raw_svector_ostream OS(Buf);
  Ctx.LS.MCIP->printInst(&DI.Inst, /*Address=*/0, /*Annot=*/"", *Ctx.LS.STI,
                         OS);
  StringRef S = StringRef(Buf).trim();
  size_t MnemEnd = S.find_first_of(" \t");
  if (MnemEnd == StringRef::npos)
    return std::nullopt;
  StringRef Rest = S.substr(MnemEnd).ltrim();
  // Operand 0 = vdst, operand 1 = src0 (matrix A).
  size_t Comma0 = Rest.find(',');
  if (Comma0 == StringRef::npos)
    return std::nullopt;
  Rest = Rest.substr(Comma0 + 1).ltrim();
  size_t Comma1 = Rest.find(',');
  StringRef A = (Comma1 == StringRef::npos) ? Rest : Rest.substr(0, Comma1);
  A = A.trim();
  if (!A.starts_with("v[") || !A.ends_with("]"))
    return std::nullopt;
  StringRef Inside = A.drop_front(2).drop_back(1);
  StringRef LoS, HiS;
  std::tie(LoS, HiS) = Inside.split(':');
  unsigned Lo = 0, Hi = 0;
  if (LoS.getAsInteger(10, Lo) || HiS.getAsInteger(10, Hi) || Hi < Lo)
    return std::nullopt;
  return VgprRange{Lo, Hi - Lo + 1};
}

// Matrix-A K-subblock masking scheme, chosen by the matrix-A data format.
// The K-split must isolate each 16-K subblock, and how a subblock maps to
// lanes/VGPRs is format-dependent:
//   * FP8/BF8: subblocks split by wave lane  -> Lane mask.
//   * FP6/BF6: subblocks split by VGPR index -> Vgpr select, 3 VGPRs/subblock.
//   * FP4    : subblocks split by VGPR index -> Vgpr select, 2 VGPRs/subblock.
enum class AMaskScheme { Lane, Vgpr };
struct AMaskPlan {
  AMaskScheme Scheme;
  unsigned SubW; // VGPRs per 16-K subblock (Vgpr scheme only)
};

// Parse "matrix_a_fmt:MATRIX_FMT_<fmt>" from the printer's canonical form and
// map it to a masking plan. FP8 is the default when the modifier is omitted.
static std::optional<AMaskPlan> matrixAMaskPlan(PatchContext &Ctx,
                                                const InternalDecodedInst &DI) {
  SmallString<256> Buf;
  raw_svector_ostream OS(Buf);
  Ctx.LS.MCIP->printInst(&DI.Inst, /*Address=*/0, /*Annot=*/"", *Ctx.LS.STI,
                         OS);
  StringRef S(Buf);
  StringRef Key = "matrix_a_fmt:MATRIX_FMT_";
  StringRef Fmt = "FP8"; // omitted modifier => default FP8
  size_t P = S.find(Key);
  if (P != StringRef::npos) {
    StringRef R = S.substr(P + Key.size());
    size_t E = R.find_first_of(" \t\r\n");
    Fmt = (E == StringRef::npos) ? R : R.substr(0, E);
  }
  if (Fmt == "FP8" || Fmt == "BF8")
    return AMaskPlan{AMaskScheme::Lane, /*SubW=*/4};
  if (Fmt == "FP6" || Fmt == "BF6")
    return AMaskPlan{AMaskScheme::Vgpr, /*SubW=*/3};
  if (Fmt == "FP4")
    return AMaskPlan{AMaskScheme::Vgpr, /*SubW=*/2};
  return std::nullopt; // unknown format -> caller fails closed
}

// Fail the whole rewrite closed rather than emit a miscompile.
static uint32_t failClosed(PatchContext &Ctx, const InternalDecodedInst &DI,
                           const Twine &Why) {
  log() << "hotswap: error: wmma_scale16: " << DI.Mnemonic << " at offset 0x"
        << utohexstr(DI.Offset) << ": " << Why
        << "; refusing to return a miscompiled code object.\n";
  Ctx.RequiredPatchFailed = true;
  return 0;
}

// ---------------------------------------------------------------------------
// v_wmma_scale16_f32_16x16x128_f8f6f4 -> exact K-split
// ---------------------------------------------------------------------------

static uint32_t patchWmmaScale16_16x16(PatchContext &Ctx, size_t Idx) {
  const InternalDecodedInst &DI = Ctx.Decoded[Idx];

  if (DI.Size != VOP3PXSize)
    return failClosed(Ctx, DI, "unexpected instruction size " + Twine(DI.Size));

  // Skip offsets a prior pass/rewrite already claimed (idempotency).
  for (const Trampoline &T : Ctx.OutTrampolines)
    if (T.OriginalOffset == DI.Offset)
      return 0;

  const uint8_t *Raw = Ctx.Text + DI.Offset;

  std::optional<unsigned> ScaleABase =
      decodeVgprEncoding(extractScaleSrc0(Raw));
  std::optional<unsigned> ScaleBBase =
      decodeVgprEncoding(extractScaleSrc1(Raw));
  if (!ScaleABase || !ScaleBBase)
    return failClosed(Ctx, DI, "non-VGPR block-16 scale operand");

  unsigned ScaleALo = *ScaleABase, ScaleAHi = ScaleALo + 1;
  unsigned ScaleBLo = *ScaleBBase, ScaleBHi = ScaleBLo + 1;

  std::optional<VgprRange> ARange = matrixAOperandRange(Ctx, DI);
  if (!ARange)
    return failClosed(Ctx, DI, "could not determine matrix-A VGPR range");
  unsigned ABase = ARange->Base;
  unsigned AWidth = ARange->Width;

  // The masking scheme depends on the matrix-A data format.
  std::optional<AMaskPlan> Plan = matrixAMaskPlan(Ctx, DI);
  if (!Plan)
    return failClosed(Ctx, DI,
                      "unrecognized matrix_a_fmt for K-subblock split");
  // For the VGPR-select scheme the 16-K subblocks must pair up (low/high)
  // across the matrix-A VGPRs; a partial trailing subblock would be malformed
  // input.
  if (Plan->Scheme == AMaskScheme::Vgpr &&
      (Plan->SubW == 0 || AWidth % (2 * Plan->SubW) != 0))
    return failClosed(Ctx, DI,
                      "matrix-A width " + Twine(AWidth) +
                          " not a multiple of subblock pair " +
                          Twine(2 * Plan->SubW));

  std::string KernelName =
      Ctx.Elf.findKernelAtAddress(DI.Offset + Ctx.Elf.textAddr());
  std::optional<unsigned> KdVgprs = Ctx.Elf.getKernelVgprCount(
      KernelName, getKernelVgprGranuleSize(Ctx, KernelName));
  unsigned KdCount = KdVgprs.value_or(Ctx.Config.MaxVgprs);

  VgprAllocator Alloc(Ctx.Liveness.LiveBefore[Idx], KdCount,
                      Ctx.Config.MaxVgprs);

  // Four block-32 scale VGPRs (A/B x low/high) plus one byte-extraction temp,
  // then a contiguous, even-aligned block for the masked A copy.
  std::optional<unsigned> ScaleAloReg = Alloc.alloc();
  std::optional<unsigned> ScaleBloReg = Alloc.alloc();
  std::optional<unsigned> ScaleAhiReg = Alloc.alloc();
  std::optional<unsigned> ScaleBhiReg = Alloc.alloc();
  std::optional<unsigned> TmpReg = Alloc.alloc();
  std::optional<unsigned> SBase =
      Alloc.allocContiguousAboveKd(AWidth, /*Align=*/2);
  if (!ScaleAloReg || !ScaleBloReg || !ScaleAhiReg || !ScaleBhiReg || !TmpReg ||
      !SBase)
    return failClosed(Ctx, DI, "insufficient scratch VGPRs for exact K-split");

  // The lane-mask scheme (FP8/BF8) needs one scratch SGPR for the wave-lane
  // bitmask; the VGPR-select scheme (FP4/FP6) uses plain v_mov and needs none.
  std::optional<SafeSgprScratchBlock> MaskSgpr;
  std::string MaskS;
  if (Plan->Scheme == AMaskScheme::Lane) {
    MaskSgpr =
        findSafeSgprScratchBlock(Ctx, DI.Offset, /*Count=*/1,
                                 /*Alignment=*/1, "wmma_scale16 lane mask");
    if (!MaskSgpr)
      return failClosed(Ctx, DI, "no scratch SGPR for lane mask");
    MaskS = ("s" + Twine(MaskSgpr->Base)).str();
  }

  // Preamble + pass-low masked copy (assembled together), then pass-high copy.
  std::string PreAsm, HiAsm;
  raw_string_ostream PreOS(PreAsm), HiOS(HiAsm);

  emitGatherEven(PreOS, vgprName(ScaleALo), vgprName(ScaleAHi),
                 vgprName(*ScaleAloReg), vgprName(*TmpReg));
  emitGatherEven(PreOS, vgprName(ScaleBLo), vgprName(ScaleBHi),
                 vgprName(*ScaleBloReg), vgprName(*TmpReg));
  emitGatherOdd(PreOS, vgprName(ScaleALo), vgprName(ScaleAHi),
                vgprName(*ScaleAhiReg), vgprName(*TmpReg));
  emitGatherOdd(PreOS, vgprName(ScaleBLo), vgprName(ScaleBHi),
                vgprName(*ScaleBhiReg), vgprName(*TmpReg));
  if (Plan->Scheme == AMaskScheme::Lane) {
    // pass-low keeps lanes 0-15 (low-16 subblocks); pass-high lanes 16-31.
    emitLaneMaskCopy(PreOS, MaskS, 0x0000FFFFu, *SBase, ABase, AWidth);
    emitLaneMaskCopy(HiOS, MaskS, 0xFFFF0000u, *SBase, ABase, AWidth);
  } else {
    // pass-low keeps the low-16 subblock VGPRs; pass-high the high-16 ones.
    emitVgprSelectCopy(PreOS, /*KeepLow=*/true, *SBase, ABase, AWidth,
                       Plan->SubW);
    emitVgprSelectCopy(HiOS, /*KeepLow=*/false, *SBase, ABase, AWidth,
                       Plan->SubW);
  }

  SmallVector<uint8_t> PreBytes = assembleInstructions(PreAsm, Ctx.LS);
  SmallVector<uint8_t> HiBytes = assembleInstructions(HiAsm, Ctx.LS);
  if (PreBytes.empty() || HiBytes.empty())
    return failClosed(Ctx, DI, "preamble assembly failed");

  // pass-low WMMA: matrix A = masked copy, scales = even-byte gathers, src2 =
  // original C (preserved by the byte copy).
  SmallVector<uint8_t> WmmaLo =
      rewriteScale16ToScale(Raw, DI.Size, VgprEncBase + *ScaleAloReg,
                            VgprEncBase + *ScaleBloReg, Ctx.LS);
  if (WmmaLo.empty())
    return failClosed(Ctx, DI, "pass-low WMMA rewrite failed");
  writeSrc0(WmmaLo.data(), VgprEncBase + *SBase);

  // pass-high WMMA: odd-byte gathers, and src2 = D so it accumulates onto the
  // pass-low result.
  SmallVector<uint8_t> WmmaHi =
      rewriteScale16ToScale(Raw, DI.Size, VgprEncBase + *ScaleAhiReg,
                            VgprEncBase + *ScaleBhiReg, Ctx.LS);
  if (WmmaHi.empty())
    return failClosed(Ctx, DI, "pass-high WMMA rewrite failed");
  writeSrc0(WmmaHi.data(), VgprEncBase + *SBase);
  writeSrc2(WmmaHi.data(), VgprEncBase + extractVdst(Raw));

  // gfx1250 WMMA co-exec hazard: the pass-high copy (VALU) overwrites the
  // masked-A block the pass-low WMMA still reads, so it must not co-execute
  // with the in-flight WMMA. Insert the full required v_nop separation between
  // them (trampoline bytes carry none of the compiler's own spacing). The
  // hazard pass re-validates each trampoline against this count as a safety
  // net.
  int A0Nops = classifyWmmaNops(DI.Mnemonic).A0Nops;
  SmallVector<uint8_t> VNop = assembleSingleInst("v_nop", Ctx.LS);
  if (VNop.empty())
    return failClosed(Ctx, DI, "v_nop assembly failed");

  SmallVector<uint8_t> Replacement;
  Replacement.append(PreBytes.begin(), PreBytes.end());
  Replacement.append(WmmaLo.begin(), WmmaLo.end());
  for (int I = 0; I < A0Nops; ++I)
    Replacement.append(VNop.begin(), VNop.end());
  Replacement.append(HiBytes.begin(), HiBytes.end());
  Replacement.append(WmmaHi.begin(), WmmaHi.end());

  unsigned Extra = Alloc.extraVgprsNeeded();
  if (checkKernelVgprBump(Ctx, KernelName, Extra, PatchRequirement::Required) !=
      VgprBumpDecision::Apply)
    return 0; // checkKernelVgprBump set RequiredPatchFailed on the Fail path.

  if (!emitToTrampoline(Ctx, DI.Offset, DI.Size, Replacement))
    return failClosed(Ctx, DI, "trampoline emission failed");

  if (MaskSgpr && !commitSafeSgprScratchBlock(Ctx, DI.Offset, *MaskSgpr,
                                              "wmma_scale16 lane mask"))
    return failClosed(Ctx, DI, "scratch SGPR commit failed");

  KernelPatchStats &Stats = Ctx.KernelStats[KernelName];
  if (Extra > Stats.ExtraVgprs)
    Stats.ExtraVgprs = Extra;
  Stats.ScratchAboveKd += Extra;

  ScratchPatchInfo Info;
  Info.Offset = DI.Offset;
  Info.ScratchRegs = Alloc.LiveAtPoint;
  Ctx.OutScratchPatches.push_back(std::move(Info));

  log() << "hotswap: wmma_scale16: exact K-split at offset 0x"
        << utohexstr(DI.Offset) << " ("
        << (Plan->Scheme == AMaskScheme::Lane ? "lane-mask" : "vgpr-select")
        << ", A=v" << ABase << ":" << (ABase + AWidth - 1) << " -> masked v"
        << *SBase << ", +" << Extra << " vgpr, " << A0Nops << " hazard v_nop, "
        << Replacement.size() << " bytes)\n";
  return 1;
}

// ---------------------------------------------------------------------------
// patchWmmaScale16 -- dispatch
// ---------------------------------------------------------------------------

static uint32_t applyWmmaScale16PatchesImpl(PatchContext &Ctx, size_t Idx) {
  StringRef Mnem(Ctx.Decoded[Idx].Mnemonic);

  if (Mnem == "v_wmma_scale16_f32_16x16x128_f8f6f4")
    return patchWmmaScale16_16x16(Ctx, Idx);

  // The M=32 FP4 form needs an M-split in addition to the K-split; not yet
  // lowered exactly, so fail closed rather than miscompile.
  if (Mnem.starts_with("v_wmma_scale16_f32_"))
    return failClosed(Ctx, Ctx.Decoded[Idx],
                      "block-16 scaled variant has no exact lowering yet");

  return 0;
}

void registerWmmaScale16Patch(HotswapPatchVTable &VT) {
  VT.applyWmmaScale16Patches = &applyWmmaScale16PatchesImpl;
}

} // namespace hotswap
} // namespace COMGR
