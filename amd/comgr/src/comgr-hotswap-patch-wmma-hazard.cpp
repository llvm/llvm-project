//===- comgr-hotswap-patch-wmma-hazard.cpp - WMMA hazard patch -----------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Whole-kernel patch for the gfx1250 WMMA/SWMMAC co-execution hazard.
/// Detects WMMA/SWMMAC instructions that lack sufficient v_nop separation
/// before the first overlapping co-executable VALU, and inserts the required
/// v_nop padding.
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"

using namespace llvm;

namespace COMGR {
namespace hotswap {
namespace {

struct WmmaHazard {
  size_t ValuIdx;
  int Deficit;
};

// Mirrors SIInstrFlags from llvm/lib/Target/AMDGPU/SIDefines.h.
// SIDefines.h is a backend-private header (not installed), so we
// duplicate the bit positions here. These must stay in sync with
// the AMDGPU backend; verify against SIDefines.h if TSFlags layout
// changes upstream.
namespace AmdgpuTSFlags {
static constexpr uint64_t VALU = UINT64_C(1) << 1;
static constexpr uint64_t IsWMMA = UINT64_C(1) << 59;
static constexpr uint64_t IsSWMMAC = UINT64_C(1) << 63;
} // namespace AmdgpuTSFlags

uint64_t getTSFlags(const MCInst &Inst, const MCInstrInfo &MCII) {
  return MCII.get(Inst.getOpcode()).TSFlags;
}

bool hasTSFlags(const MCInst &Inst, const MCInstrInfo &MCII, uint64_t Mask) {
  return (getTSFlags(Inst, MCII) & Mask) != 0;
}

bool isWmmaLike(const MCInst &Inst, const MCInstrInfo &MCII) {
  return hasTSFlags(Inst, MCII,
                    AmdgpuTSFlags::IsWMMA | AmdgpuTSFlags::IsSWMMAC);
}

bool isVNop(const InternalDecodedInst &DI) { return DI.Mnemonic == "v_nop"; }

bool isCoexecutableVALU(const InternalDecodedInst &DI,
                        const MCInstrInfo &MCII) {
  if (isVNop(DI))
    return false;
  if (!hasTSFlags(DI.Inst, MCII, AmdgpuTSFlags::VALU))
    return false;
  return !isWmmaLike(DI.Inst, MCII);
}

bool isTerminatingSalu(const MCInst &Inst, const MCInstrInfo &MCII) {
  const MCInstrDesc &Desc = MCII.get(Inst.getOpcode());
  return Desc.isTerminator() || Desc.isBranch() || Desc.isCall() ||
         Desc.isReturn();
}

} // anonymous namespace

// Checks are ordered most-restrictive-first. If a mnemonic matches
// multiple substrings (e.g. contains both "_iu8" and "_f16"), the
// first match wins. Do not reorder without verifying the required nop counts.
WmmaNopReq classifyWmmaNops(StringRef Mnemonic) {
  // Redundant in production (caller filters via isWmmaLike), but kept
  // as a defensive guard since classifyWmmaNops is a public function
  // also exercised directly by unit tests with non-WMMA mnemonics.
  bool IsWmma = Mnemonic.starts_with("v_wmma");
  bool IsSwmmac = Mnemonic.starts_with("v_swmmac");
  if (!IsWmma && !IsSwmmac)
    return {4, 4};

  if (Mnemonic.contains("_iu8") || Mnemonic.contains("_iu4"))
    return {8, 4};

  if (Mnemonic.contains("f8f6f4"))
    return {1, 4};

  if (Mnemonic.contains("_fp8") || Mnemonic.contains("_f8") ||
      Mnemonic.contains("_bf8")) {
    if (Mnemonic.contains("16x16x128"))
      return {3, 4};
    return {1, 4};
  }

  if (Mnemonic.contains("_f16") || Mnemonic.contains("_bf16"))
    return {4, 4};

  return {4, 4};
}

namespace {

// Scan a decoded stream for WMMA/SWMMAC -> overlapping co-executable VALU
// hazards, returning the nop deficits keyed by VALU index (into \p Stream).
//
// \p RequireAbsolute picks the nop budget. The original kernel stream (false)
// already carries the compiler-inserted spacing, so only the extra deficit
// beyond that matters; freshly emitted trampoline bodies (true) have no such
// baseline, so the full required count is enforced.
std::vector<WmmaHazard> scanCoexecHazards(ArrayRef<InternalDecodedInst> Stream,
                                          const MCInstrInfo &MCII,
                                          const MCRegisterInfo &MRI,
                                          bool RequireAbsolute,
                                          int *WmmaScannedOut = nullptr) {
  std::vector<WmmaHazard> Hazards;
  DenseSet<size_t> PatchedValuIndices;
  int WmmaScanned = 0;

  for (size_t WmmaIdx = 0, E = Stream.size(); WmmaIdx < E; ++WmmaIdx) {
    const InternalDecodedInst &WmmaDI = Stream[WmmaIdx];
    if (!isWmmaLike(WmmaDI.Inst, MCII))
      continue;

    ++WmmaScanned;
    WmmaNopReq Req = classifyWmmaNops(WmmaDI.Mnemonic);
    // Target is always the full required count. For the original stream, the
    // compiler-inserted baseline only gates whether a scan is needed: if the
    // requirement is no more than that baseline, the existing spacing suffices.
    // Trampoline bodies have no baseline, so the full count is always enforced.
    if (!RequireAbsolute && Req.A0Nops <= Req.B0Nops)
      continue;
    int Target = Req.A0Nops;

    int SafeSlots = 0;
    for (size_t ValuIdx = WmmaIdx + 1; ValuIdx < E; ++ValuIdx) {
      const InternalDecodedInst &Candidate = Stream[ValuIdx];

      if (isVNop(Candidate)) {
        ++SafeSlots;
        if (SafeSlots >= Target)
          break;
        continue;
      }

      if (!hasTSFlags(Candidate.Inst, MCII, AmdgpuTSFlags::VALU)) {
        if (isTerminatingSalu(Candidate.Inst, MCII))
          break;
        continue;
      }

      if (isCoexecutableVALU(Candidate, MCII)) {
        if (!checkVgprOverlap(WmmaDI.Inst, Candidate.Inst, MRI)) {
          ++SafeSlots;
          if (SafeSlots >= Target)
            break;
          continue;
        }

        if (SafeSlots < Target && PatchedValuIndices.insert(ValuIdx).second) {
          Hazards.push_back({ValuIdx, Target - SafeSlots});
          log() << "hotswap: WMMA co-exec hazard at 0x"
                << utohexstr(WmmaDI.Offset) << ": " << WmmaDI.Mnemonic
                << " needs " << Target << " v_nops, only " << SafeSlots
                << " found before " << Candidate.Mnemonic << " at 0x"
                << utohexstr(Candidate.Offset) << "\n";
        }
        break;
      }

      break;
    }
  }

  if (WmmaScannedOut)
    *WmmaScannedOut = WmmaScanned;
  return Hazards;
}

std::vector<WmmaHazard> findWmmaCoexecHazards(const PatchContext &Ctx) {
  int WmmaScanned = 0;
  std::vector<WmmaHazard> Hazards =
      scanCoexecHazards(Ctx.Decoded, *Ctx.LS.MCII, *Ctx.LS.MRI,
                        /*RequireAbsolute=*/false, &WmmaScanned);
  log() << "hotswap: WMMA co-exec validation: " << Hazards.size()
        << " hazards (" << WmmaScanned << " WMMA instructions scanned)\n";
  return Hazards;
}

// Fail-closed safety net for trampoline bodies emitted by other passes (e.g.
// the exact scale16 K-split): they contain their own WMMAs but were never in
// Ctx.Decoded, so findWmmaCoexecHazards cannot see them, and they carry no
// compiler-inserted spacing, so each is checked against the full required
// count. A residual deficit means a pass shipped an unsafe WMMA/VALU ordering,
// so we refuse the rewrite. Runs before branch fixup/coalescing, so each
// T.Bytes is still {body, reserved-return-slot}; the reserved tail is trimmed
// before decoding since its zero bytes are not real instructions.
void validateTrampolineCoexec(PatchContext &Ctx) {
  const MCInstrInfo &MCII = *Ctx.LS.MCII;
  const MCRegisterInfo &MRI = *Ctx.LS.MRI;

  for (const Trampoline &T : Ctx.OutTrampolines) {
    unsigned Reserve = T.Long ? SetPcReturnReserveBytes : MinInstSize;
    if (T.Bytes.size() <= Reserve)
      continue;
    size_t BodySize = T.Bytes.size() - Reserve;

    std::vector<InternalDecodedInst> Body;
    if (!decodeTextSection(T.Bytes.data(), BodySize, Ctx.LS, Body))
      continue;

    std::vector<WmmaHazard> TH =
        scanCoexecHazards(Body, MCII, MRI, /*RequireAbsolute=*/true);
    if (!TH.empty()) {
      log() << "hotswap: error: WMMA co-exec hazard unmitigated in trampoline "
               "for site 0x"
            << utohexstr(T.OriginalOffset) << " (" << TH.size()
            << " site(s)); failing closed\n";
      Ctx.RequiredPatchFailed = true;
    }
  }
}

} // anonymous namespace

static uint32_t applyWmmaHazardPatchImpl(PatchContext &Ctx) {
  std::vector<WmmaHazard> Hazards = findWmmaCoexecHazards(Ctx);
  if (Hazards.empty()) {
    // Even with no original-stream hazards, other passes may have emitted
    // WMMA-bearing trampolines that need their own separation validated.
    validateTrampolineCoexec(Ctx);
    return 0;
  }

  uint32_t Patched = 0;
  for (const WmmaHazard &H : Hazards) {
    const InternalDecodedInst &ValuDI = Ctx.Decoded[H.ValuIdx];

    uint64_t TrampolineTextOffset = Ctx.TextSize;
    for (const Trampoline &T : Ctx.OutTrampolines)
      TrampolineTextOffset += T.Bytes.size();

    SmallVector<MCInst> Insts;
    for (int I = 0; I < H.Deficit; ++I)
      Insts.push_back(Ctx.LS.VNopInst);
    Insts.push_back(ValuDI.Inst);

    Trampoline T = buildTrampoline(Insts, ValuDI.Offset, ValuDI.Size,
                                   TrampolineTextOffset, Ctx.LS);
    if (T.Bytes.empty()) {
      log() << "hotswap: error: WMMA hazard: buildTrampoline failed at 0x"
            << utohexstr(ValuDI.Offset) << "\n";
      continue;
    }
    Ctx.OutTrampolines.push_back(std::move(T));

    log() << "hotswap: WMMA hazard fix at 0x" << utohexstr(ValuDI.Offset)
          << ": inserted " << H.Deficit << " v_nop(s)\n";
    ++Patched;
  }

  // Validate every emitted trampoline (including the ones just added above and
  // any WMMA-bearing bodies from other passes) against the full required
  // budget.
  validateTrampolineCoexec(Ctx);
  return Patched;
}

void registerWmmaHazardPatch(HotswapPatchVTable &VT) {
  VT.applyWmmaHazardPatch = &applyWmmaHazardPatchImpl;
}

} // namespace hotswap
} // namespace COMGR
