//===- comgr-hotswap-patch-inplace.cpp - In-place B0-to-A0 patches --------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Strong-symbol override for applyInPlacePatches.  Handles instruction
/// rewrites that fit in the same code size as the original:
///
///   - s_clause                 -> s_nop 0
///   - cluster_load             -> global_load    (opcode swap via MCInst +
///                                                 MCCodeEmitter)
///   - s_barrier_signal_isfirst -> s_barrier_signal
///                                                (opcode swap; same operand
///                                                 layout, drops SCC write)
///
/// No trampolines, ELF growth, or extra VGPRs are required.
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCRegisterInfo.h"

using namespace llvm;

namespace COMGR {
namespace hotswap {
namespace {

/// Map a B0-only cluster_load mnemonic to the assembly string of its
/// A0-compatible global_load equivalent (with a dummy operand to resolve
/// the opcode). Returns an empty StringRef if \p Mnemonic is not a
/// cluster_load variant.
StringRef getClusterLoadReplacementAsm(StringRef Mnemonic) {
  return StringSwitch<StringRef>(Mnemonic)
      .Case("cluster_load_b32", "global_load_b32 v0, v[0:1], off")
      .Case("cluster_load_b64", "global_load_b64 v[0:1], v[2:3], off")
      .Case("cluster_load_b128", "global_load_b128 v[0:3], v[4:5], off")
      .Case("cluster_load_async_to_lds_b8",
            "global_load_async_to_lds_b8 v0, v[0:1], off")
      .Case("cluster_load_async_to_lds_b32",
            "global_load_async_to_lds_b32 v0, v[0:1], off")
      .Case("cluster_load_async_to_lds_b64",
            "global_load_async_to_lds_b64 v0, v[0:1], off")
      .Case("cluster_load_async_to_lds_b128",
            "global_load_async_to_lds_b128 v0, v[0:1], off")
      .Default("");
}

/// Detect the SGPR-relative (_SADDR) form of a cluster_load.
///
/// The off-form cluster_load carries its global address in a 64-bit VGPR pair
/// and encodes the saddr field as the "off" sentinel, so all of its register
/// operands are VGPRs. The _SADDR variant shares the same display mnemonic but
/// is a distinct MC opcode with an extra scalar saddr operand. The off-form
/// replacement templates in getClusterLoadReplacementAsm would mis-encode that
/// scalar operand, so the two forms must be told apart. getNamedOperandIdx() /
/// OpName are backend-private headers, so -- mirroring the operand-kind
/// inspection in comgr-hotswap-patch-wmma-split.cpp -- classify by operand
/// kind: the presence of any SGPR register operand marks the _SADDR form.
bool usesSgprBaseAddress(const MCInst &Inst, const MCRegisterInfo &MRI) {
  for (unsigned I = 0, E = Inst.getNumOperands(); I < E; ++I) {
    const MCOperand &Op = Inst.getOperand(I);
    if (Op.isReg() && Op.getReg() &&
        StringRef(MRI.getName(Op.getReg())).starts_with("SGPR"))
      return true;
  }
  return false;
}

/// Resolve the MC opcode index for an assembly mnemonic by parsing a dummy
/// instruction through the asm parser.
std::optional<unsigned> resolveOpcode(StringRef AsmSnippet,
                                      const LLVMState &LS) {
  SmallVector<uint8_t> Bytes = assembleSingleInst(AsmSnippet, LS);
  if (Bytes.empty())
    return std::nullopt;
  std::vector<InternalDecodedInst> Decoded;
  if (!decodeTextSection(Bytes.data(), Bytes.size(), LS, Decoded) ||
      Decoded.empty())
    return std::nullopt;
  return Decoded[0].Inst.getOpcode();
}

/// Encode an MCInst to raw bytes via MCCodeEmitter.
SmallVector<uint8_t> encodeMCInst(const MCInst &Inst, const LLVMState &LS) {
  SmallVector<char, 16> Code;
  SmallVector<MCFixup, 4> Fixups;
  LS.MCE->encodeInstruction(Inst, Code, Fixups, *LS.STI);
  return SmallVector<uint8_t>(Code.begin(), Code.end());
}

/// Perform an opcode swap: clone the decoded MCInst, set the replacement
/// opcode, re-encode via MCCodeEmitter, and overwrite in place.
/// Returns true on success.
bool swapOpcode(InternalDecodedInst &DI, uint8_t *Text, const LLVMState &LS,
                unsigned NewOpcode) {
  MCInst NewInst = DI.Inst;
  NewInst.setOpcode(NewOpcode);
  SmallVector<uint8_t> Bytes = encodeMCInst(NewInst, LS);
  if (Bytes.empty() || Bytes.size() != DI.Size)
    return false;
  std::memcpy(Text + DI.Offset, Bytes.data(), DI.Size);
  return true;
}

} // anonymous namespace

static uint32_t applyInPlacePatchesImpl(PatchContext &Ctx, size_t Idx) {
  InternalDecodedInst &DI = Ctx.Decoded[Idx];
  StringRef Mnemonic(DI.Mnemonic);

  if (DI.Inst.getOpcode() == Ctx.LS.SClauseOpcode) {
    std::memcpy(Ctx.Text + DI.Offset, Ctx.LS.SNopBytes.data(),
                Ctx.LS.SNopBytes.size());
    log() << "hotswap: inplace: s_clause -> s_nop 0 at 0x"
          << utohexstr(DI.Offset) << "\n";
    return 1;
  }

  StringRef ReplacementAsm = getClusterLoadReplacementAsm(Mnemonic);
  if (!ReplacementAsm.empty()) {
    HotswapProfile::Scope S =
        Ctx.Profile.time(HotswapMetric::InPlaceClusterLoad);
    // The replacement templates above are all the saddr=off encoding form
    // (global address in a 64-bit VGPR pair). The SGPR-relative (_SADDR)
    // cluster_load shares the mnemonic but has a different operand layout, so
    // reusing the off-form opcode would re-encode its scalar saddr and 32-bit
    // vaddr as a 64-bit vaddr plus an inline offset -- a corrupt address that
    // faults the GPU at runtime. Leave the _SADDR form unchanged here so the
    // later trampoline pass can preserve the cluster-load opcode and wrap it
    // with the A0-required M0 wg_mask clear/restore sequence.
    if (usesSgprBaseAddress(DI.Inst, *Ctx.LS.MRI)) {
      log() << "hotswap: inplace: " << Mnemonic
            << " (SGPR-relative saddr form) left unchanged at 0x"
            << utohexstr(DI.Offset) << "\n";
    } else {
      std::optional<unsigned> NewOpcode = resolveOpcode(ReplacementAsm, Ctx.LS);
      if (NewOpcode && swapOpcode(DI, Ctx.Text, Ctx.LS, *NewOpcode)) {
        log() << "hotswap: inplace: " << Mnemonic << " -> opcode " << *NewOpcode
              << " at 0x" << utohexstr(DI.Offset) << "\n";
        S.addPatches(1);
        return 1;
      }
    }
  }

  // s_barrier_signal_isfirst -> s_barrier_signal: on A0, the isfirst
  // variant may return stale SCC when cluster barriers are in flight.
  // Both S_BARRIER_SIGNAL_IMM and S_BARRIER_SIGNAL_ISFIRST_IMM share
  // a single SplitBarrier:$src0 immediate operand (see SOPInstructions.td),
  // so cloning the decoded MCInst and flipping the opcode preserves the
  // original barrier-ID operand. The dummy "-1" is only used to resolve
  // the target opcode via the asm parser.
  //
  // Correctness caveat: the isfirst variant defines SCC; the non-isfirst
  // variant does not. If downstream code reads SCC expecting the result
  // of isfirst (e.g. an s_cbranch_scc1 selecting the elected wave), the
  // swap leaves that read consuming stale SCC. On A0 the isfirst result
  // is already unreliable due to the underlying race, so the swap removes
  // a known-broken code path rather than introducing a new one. But it
  // is not a semantic equivalence. Liveness/CFG-aware detection of SCC
  // consumers is undecidable in general; the proper fix lives in
  // A0-targeted Clang codegen and is out of scope for hotswap. This
  // patch is a runtime mitigation for B0 binaries running on A0.
  //
  // The _M0 form has a different tablegen mnemonic string
  // ("s_barrier_signal_isfirst m0", with the "m0" baked into the
  // mnemonic itself, not as an operand -- see S_BARRIER_SIGNAL_ISFIRST_M0
  // in SOPInstructions.td), so it does not match this equality check
  // and falls through to the dispatcher's "no match" return below.
  // The AMDGPU backend never emits the _M0 form for compute kernels.
  if (Mnemonic == "s_barrier_signal_isfirst") {
    HotswapProfile::Scope S =
        Ctx.Profile.time(HotswapMetric::InPlaceBarrierSignal);
    std::optional<unsigned> NewOpcode =
        resolveOpcode("s_barrier_signal -1", Ctx.LS);
    if (NewOpcode && swapOpcode(DI, Ctx.Text, Ctx.LS, *NewOpcode)) {
      log() << "hotswap: inplace: s_barrier_signal_isfirst -> opcode "
            << *NewOpcode << " at 0x" << utohexstr(DI.Offset) << "\n";
      S.addPatches(1);
      return 1;
    }
  }

  return 0;
}

void registerInPlacePatch(HotswapPatchVTable &VT) {
  VT.applyInPlacePatches = &applyInPlacePatchesImpl;
}

} // namespace hotswap
} // namespace COMGR
