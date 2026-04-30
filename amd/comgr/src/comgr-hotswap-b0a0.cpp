//===- comgr-hotswap-b0a0.cpp - GFX1250 B0-to-A0 patch dispatcher --------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Dispatcher for B0-to-A0 silicon stepping patches and the
/// retargetCodeObjectB0A0 orchestrator that drives the full pipeline:
/// decode -> patch -> trampoline growth -> DWARF update.
///
/// Patch entry points are declared as weak symbols returning 0. Each
/// comgr-hotswap-patch-*.cpp file provides a strong override, allowing
/// patches to land as independent PRs with no merge conflicts.
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Compiler.h"

using namespace llvm;

namespace COMGR {
namespace hotswap {

// -- GFX1250 B0-to-A0 constants -----------------------------------------------
//
// All instruction encoding lives in LLVMState (s_branch opcode + pre-encoded
// s_nop bytes, populated at initLLVM time via the MC asm parser). This policy
// layer only carries ISA identifiers and register granularity -- no
// target-specific opcode bits should land here.

static constexpr unsigned Gfx1250MaxVgprs = 256;
// GFX12 wave32 VGPR granularity; SGPR granularity is a fixed 16 across all
// AMDGPU generations Comgr's hotswap currently supports.
static constexpr unsigned Gfx1250VgprGranuleSize = 8;
static constexpr unsigned Gfx1250SgprGranuleSize = 16;

/// Build the default RewriteConfig used for the GFX1250 B0-to-A0 rewrite:
/// fills in the identity source / target ISA (both gfx1250) and the
/// AMDGPU register granularity constants consumed by
/// ElfView::updateKernelDescriptor. Instruction-encoding state is not
/// carried in RewriteConfig; see LLVMState for the s_branch opcode and
/// pre-encoded s_nop bytes.
static RewriteConfig makeGfx1250B0A0Config() {
  // `Config` / `Cfg` are reserved below: `Config` always names a
  // RewriteConfig; `Cfg` is only used for the CFG (control-flow graph)
  // local in applyGfx1250B0toA0Rules.
  RewriteConfig Config;
  Config.SourceIsa = "amdgcn-amd-amdhsa--gfx1250";
  Config.TargetIsa = "amdgcn-amd-amdhsa--gfx1250";
  Config.TargetCpu = "gfx1250";
  Config.MaxVgprs = Gfx1250MaxVgprs;
  Config.VgprGranuleSize = Gfx1250VgprGranuleSize;
  Config.SgprGranuleSize = Gfx1250SgprGranuleSize;
  return Config;
}

// -- Forward declarations for patch/liveness/DWARF stubs ----------------------
//
// These have weak default definitions below; patch .cpp files may provide
// strong overrides at link time so patches can land as independent PRs.

uint32_t applyInPlacePatches(PatchContext &, size_t);
uint32_t applyTrampolinePatches(PatchContext &, size_t);
uint32_t applyWmmaHazardPatch(PatchContext &);
uint32_t applyWmmaSplitPatches(PatchContext &, size_t);
uint32_t applyScratchPatches(PatchContext &, size_t);
CFG buildCfg(ArrayRef<InternalDecodedInst> Decoded, const MCInstrInfo &);
LivenessInfo computeLiveness(ArrayRef<InternalDecodedInst> Decoded, const CFG &,
                             const MCInstrInfo &, const MCRegisterInfo &,
                             unsigned MaxVgprs);
RegDefUse getInstRegDefUse(const MCInst &, const MCInstrInfo &,
                           const MCRegisterInfo &);
int64_t getBranchImm(const MCInst &);
bool verifyPatchCorrectness(const uint8_t *, uint64_t, const LLVMState &,
                            ArrayRef<ScratchPatchInfo>, unsigned);
bool addTrampolineSymbols(WritableMemoryBuffer &ElfBuf,
                          ArrayRef<Trampoline> Trampolines,
                          uint64_t TextSizeBefore, unsigned TextSectionIdx);
bool patchDebugLine(WritableMemoryBuffer &ElfBuf,
                    ArrayRef<Trampoline> Trampolines, uint64_t TextSizeBefore,
                    uint64_t TextAddr);
void patchDebugRanges(uint8_t *Elf, size_t ElfSize, uint64_t TextAddr,
                      uint64_t TextSizeBefore, uint64_t TrampTotal);
void patchDebugInfo(uint8_t *Elf, size_t ElfSize, uint64_t TextAddr,
                    uint64_t TextSizeBefore, uint64_t TrampTotal);
void patchDebugFrame(uint8_t *Elf, size_t ElfSize, uint64_t TextAddr,
                     uint64_t TextSizeBefore, uint64_t TrampTotal);

// -- Weak-symbol patch stubs --------------------------------------------------

LLVM_ATTRIBUTE_WEAK uint32_t applyInPlacePatches(PatchContext &, size_t) {
  return 0;
}
LLVM_ATTRIBUTE_WEAK uint32_t applyTrampolinePatches(PatchContext &, size_t) {
  return 0;
}
LLVM_ATTRIBUTE_WEAK uint32_t applyWmmaHazardPatch(PatchContext &) { return 0; }
LLVM_ATTRIBUTE_WEAK uint32_t applyWmmaSplitPatches(PatchContext &, size_t) {
  return 0;
}
LLVM_ATTRIBUTE_WEAK uint32_t applyScratchPatches(PatchContext &, size_t) {
  return 0;
}

// -- Weak-symbol liveness stubs -----------------------------------------------
//
// Conservative defaults: all VGPRs reported live. ScratchAllocator will
// allocate above KD count (correct but suboptimal until the real liveness
// layer lands).

LLVM_ATTRIBUTE_WEAK CFG buildCfg(ArrayRef<InternalDecodedInst> Decoded,
                                 const MCInstrInfo &) {
  (void)Decoded;
  return CFG();
}

LLVM_ATTRIBUTE_WEAK LivenessInfo computeLiveness(
    ArrayRef<InternalDecodedInst> Decoded, const CFG &, const MCInstrInfo &,
    const MCRegisterInfo &, unsigned MaxVgprs) {
  LivenessInfo Info;
  BitVector AllLive(MaxVgprs);
  AllLive.set(0, MaxVgprs);
  Info.LiveBefore.resize(Decoded.size(), AllLive);
  Info.LiveAfter.resize(Decoded.size(), AllLive);
  Info.Converged = true;
  return Info;
}

LLVM_ATTRIBUTE_WEAK RegDefUse getInstRegDefUse(const MCInst &,
                                               const MCInstrInfo &,
                                               const MCRegisterInfo &) {
  return {};
}

LLVM_ATTRIBUTE_WEAK int64_t getBranchImm(const MCInst &) { return 0; }

LLVM_ATTRIBUTE_WEAK bool verifyPatchCorrectness(const uint8_t *, uint64_t,
                                                const LLVMState &,
                                                ArrayRef<ScratchPatchInfo>,
                                                unsigned) {
  return true;
}

// -- Weak-symbol DWARF stubs --------------------------------------------------

LLVM_ATTRIBUTE_WEAK bool addTrampolineSymbols(WritableMemoryBuffer &,
                                              ArrayRef<Trampoline>, uint64_t,
                                              unsigned) {
  return true;
}
LLVM_ATTRIBUTE_WEAK bool patchDebugLine(WritableMemoryBuffer &,
                                        ArrayRef<Trampoline>, uint64_t,
                                        uint64_t) {
  return true;
}
LLVM_ATTRIBUTE_WEAK void patchDebugRanges(uint8_t *, size_t, uint64_t, uint64_t,
                                          uint64_t) {}
LLVM_ATTRIBUTE_WEAK void patchDebugInfo(uint8_t *, size_t, uint64_t, uint64_t,
                                        uint64_t) {}
LLVM_ATTRIBUTE_WEAK void patchDebugFrame(uint8_t *, size_t, uint64_t, uint64_t,
                                         uint64_t) {}

// -- NOP sled scanning --------------------------------------------------------

/// Scan \p Decoded for runs of consecutive `s_nop` instructions at least
/// MinNopSledSize bytes long and return the resulting NopSled list (each
/// sled records Start / End byte offsets in .text and the initial WritePos
/// at Start). These sleds are the landing zones emitToNopSled targets for
/// in-place rewrites. NOPs are identified by MC opcode (cached on \p LS at
/// initLLVM() time) rather than mnemonic string, so the scanner is robust
/// against printer aliasing / mnemonic formatting variations.
static std::vector<NopSled>
buildNopSledMap(ArrayRef<InternalDecodedInst> Decoded, const LLVMState &LS) {
  std::vector<NopSled> Sleds;
  const size_t N = Decoded.size();
  size_t I = 0;
  while (I < N) {
    if (Decoded[I].Inst.getOpcode() == LS.SNopOpcode) {
      uint64_t Start = Decoded[I].Offset;
      uint64_t End = Start;
      while (I < N && Decoded[I].Inst.getOpcode() == LS.SNopOpcode) {
        End = Decoded[I].Offset + Decoded[I].Size;
        ++I;
      }
      if (End - Start >= MinNopSledSize)
        Sleds.push_back({Start, End, Start});
    } else {
      ++I;
    }
  }
  return Sleds;
}

// -- Sled-or-trampoline code emission -----------------------------------------

/// Emit the replacement code for the instruction at [\p InstOffset,
/// \p InstOffset + \p InstSize) into a nearby NOP sled: writes \p Replacement
/// into the sled, appends a branch-back to the next instruction after the
/// original site, overwrites the original site with a branch-forward to the
/// sled, and pads the leftover bytes of the original slot with cached s_nop
/// bytes. Advances \c Sled.WritePos by the amount consumed. Returns false if
/// either branch encoding fails, leaving \c Ctx.Text partially written.
[[nodiscard]] static bool emitToNopSled(PatchContext &Ctx, NopSled &Sled,
                                        uint64_t InstOffset, uint32_t InstSize,
                                        ArrayRef<uint8_t> Replacement) {
  const LLVMState &LS = Ctx.LS;
  std::memcpy(Ctx.Text + Sled.WritePos, Replacement.data(), Replacement.size());

  SmallVector<uint8_t> BrBack = LS.encodeSBranch(
      Sled.WritePos + Replacement.size(), InstOffset + InstSize);
  if (BrBack.empty()) {
    log() << "hotswap: error: emitToNopSled: encodeSBranch for branch-back "
          << "at sled offset 0x"
          << utohexstr(Sled.WritePos + Replacement.size()) << " -> 0x"
          << utohexstr(InstOffset + InstSize) << " failed.\n";
    return false;
  }
  std::memcpy(Ctx.Text + Sled.WritePos + Replacement.size(), BrBack.data(),
              BrBack.size());

  SmallVector<uint8_t> BrFwd = LS.encodeSBranch(InstOffset, Sled.WritePos);
  if (BrFwd.empty()) {
    log() << "hotswap: error: emitToNopSled: encodeSBranch for branch-fwd "
          << "at original offset 0x" << utohexstr(InstOffset) << " -> sled 0x"
          << utohexstr(Sled.WritePos) << " failed.\n";
    return false;
  }
  std::memcpy(Ctx.Text + InstOffset, BrFwd.data(), BrFwd.size());

  // Pad the tail of the replaced instruction slot with cached s_nop bytes
  // (pre-encoded in LLVMState at initLLVM() time).
  for (uint32_t I = MinInstSize; I < InstSize; I += MinInstSize)
    std::memcpy(Ctx.Text + InstOffset + I, LS.SNopBytes.data(), MinInstSize);

  Sled.WritePos += Replacement.size() + MinInstSize;
  return true;
}

/// Queue a deferred trampoline for the instruction at [\p InstOffset,
/// \p InstOffset + \p InstSize) with \p Replacement as its body. The final
/// branch encoding (branch-back at the trampoline tail and branch-forward
/// overwrite at the original site) is filled in by fixupTrampolineBranches
/// once the post-.text trampoline layout is known -- we reserve
/// MinInstSize zero bytes at the end of the trampoline body as a
/// placeholder rather than encoding twice. Used when there is no reachable
/// NOP sled for an in-place sled patch.
[[nodiscard]] static bool emitToTrampoline(PatchContext &Ctx,
                                           uint64_t InstOffset,
                                           uint32_t InstSize,
                                           ArrayRef<uint8_t> Replacement) {
  Trampoline T;
  T.OriginalOffset = InstOffset;
  T.OriginalSize = InstSize;
  T.Bytes.insert(T.Bytes.end(), Replacement.begin(), Replacement.end());
  // Reserve the branch-back slot; fixupTrampolineBranches fills it in.
  T.Bytes.insert(T.Bytes.end(), MinInstSize, uint8_t{0});
  Ctx.OutTrampolines.emplace_back(std::move(T));
  return true;
}

/// Emit \p Replacement for the instruction at [\p InstOffset,
/// \p InstOffset + \p InstSize). Prefers an in-place NOP-sled rewrite when a
/// reachable sled with sufficient headroom exists; otherwise falls back to a
/// deferred trampoline. Marked [[maybe_unused]] because the weak-stub patch
/// passes in this file do not yet call it -- the concrete patch .cpp files
/// that land alongside will.
[[maybe_unused, nodiscard]] static bool
emitReplacementCode(PatchContext &Ctx, uint64_t InstOffset, uint32_t InstSize,
                    ArrayRef<uint8_t> Replacement) {
  // findNearestSled already enforces that the returned sled has at least
  // `Needed` bytes of headroom, so a non-null result is sufficient to take
  // the in-place path.
  uint64_t Needed = Replacement.size() + MinInstSize;
  if (NopSled *Sled = findNearestSled(Ctx.NopSleds, InstOffset, Needed))
    return emitToNopSled(Ctx, *Sled, InstOffset, InstSize, Replacement);
  return emitToTrampoline(Ctx, InstOffset, InstSize, Replacement);
}

// -- applyGfx1250B0toA0Rules --------------------------------------------------

/// Main per-instruction dispatcher for the GFX1250 B0-to-A0 rewrite.
/// Builds the NOP sled map, CFG, and VGPR liveness for the decoded stream,
/// then walks each decoded instruction and runs the patch passes in order
/// (in-place -> trampoline -> WMMA split -> scratch). Each pass gets a
/// chance to claim the instruction; first non-zero return wins. Also runs
/// the whole-function WMMA-hazard pass after the per-instruction loop and
/// records per-kernel stats via ElfView::updateKernelDescriptor.
/// Returns the total number of applied patches across all passes.
static uint32_t
applyGfx1250B0toA0Rules(std::vector<InternalDecodedInst> &Decoded,
                        uint8_t *Text, uint64_t TextSize, const LLVMState &LS,
                        std::vector<Trampoline> &OutTrampolines, ElfView &Elf,
                        std::vector<ScratchPatchInfo> &OutScratchPatches,
                        const RewriteConfig &Config) {
  uint32_t Patched = 0;
  std::vector<NopSled> Sleds = buildNopSledMap(Decoded, LS);

  CFG Cfg = buildCfg(Decoded, *LS.MCII);
  LivenessInfo Liveness =
      computeLiveness(Decoded, Cfg, *LS.MCII, *LS.MRI, Config.MaxVgprs);

  if (!Liveness.Converged) {
    log() << "hotswap: error: liveness analysis did not converge, using "
          << "conservative all-VGPRs-live fallback\n";
    BitVector AllVgprs(Config.MaxVgprs);
    AllVgprs.set(0, Config.MaxVgprs);
    for (size_t I = 0, LE = Liveness.LiveBefore.size(); I < LE; ++I) {
      Liveness.LiveBefore[I] = AllVgprs;
      Liveness.LiveAfter[I] = AllVgprs;
    }
  }

  StringMap<KernelPatchStats> KernelStats;
  PatchContext Ctx{Config,           Decoded, Text, TextSize, LS,
                   OutTrampolines,   Sleds,   Elf,  Liveness, KernelStats,
                   OutScratchPatches};

  for (size_t Idx = 0, E = Decoded.size(); Idx < E; ++Idx) {
    const InternalDecodedInst &DI = Decoded[Idx];
    if (DI.Mnemonic == "<unknown>")
      continue;

    uint32_t P = 0;
    P += applyInPlacePatches(Ctx, Idx);
    if (P) {
      Patched += P;
      continue;
    }
    P += applyTrampolinePatches(Ctx, Idx);
    if (P) {
      Patched += P;
      continue;
    }
    P += applyWmmaSplitPatches(Ctx, Idx);
    if (P) {
      Patched += P;
      continue;
    }
    P += applyScratchPatches(Ctx, Idx);
    if (P) {
      Patched += P;
      continue;
    }
  }

  // The WMMA hazard pass runs after per-instruction patches. Earlier passes
  // may have modified Text bytes, but the Decoded stream still holds the
  // original MCInst/Mnemonic/Offset entries. This is safe because:
  //  - In-place patches only change opcodes within the same encoding size,
  //    preserving instruction boundaries and offsets.
  //  - Trampoline patches replace the original instruction with a branch
  //    (same size), so the Decoded entry's Offset still points at the
  //    branch site, the WMMA classifier won't match a branch as WMMA/VALU.
  // If a future patch family changes instruction boundaries, the Decoded
  // stream must be rebuilt before this pass runs.
  Patched += applyWmmaHazardPatch(Ctx);

  for (const llvm::StringMapEntry<KernelPatchStats> &KV : KernelStats) {
    StringRef KName = KV.first();
    const KernelPatchStats &Stats = KV.second;
    if (KName.empty())
      continue;
    std::optional<unsigned> VgprsBefore =
        Elf.getKernelVgprCount(KName, Config.VgprGranuleSize);
    if (Stats.ExtraVgprs > 0)
      Elf.updateKernelDescriptor(KName, Stats.ExtraVgprs, 0,
                                 Config.VgprGranuleSize,
                                 Config.SgprGranuleSize);
    std::optional<unsigned> VgprsAfter =
        Elf.getKernelVgprCount(KName, Config.VgprGranuleSize);
    log() << "hotswap: liveness: kernel " << KName
          << ": vgprs_before=" << VgprsBefore.value_or(0)
          << ", vgprs_after=" << VgprsAfter.value_or(0)
          << ", scratch_reused=" << Stats.ScratchReused
          << ", scratch_above_kd=" << Stats.ScratchAboveKd << "\n";
  }
  return Patched;
}

// -- retargetCodeObjectB0A0 helpers -------------------------------------------

/// Finalize the deferred trampolines produced by emitToTrampoline: resolves
/// the branch-back at the tail of each trampoline to land on the next
/// instruction after the original site, writes the branch-forward + s_nop
/// padding at the original .text slot, and reports per-trampoline encoding
/// failures through log(). Runs after all patch passes finish so the
/// post-.text layout of trampolines is known. Returns false if any
/// trampoline could not be fixed up, but still patches the ones that can.
[[nodiscard]] static bool
fixupTrampolineBranches(std::vector<Trampoline> &Trampolines, uint8_t *Text,
                        uint64_t TextSize, const LLVMState &LS) {
  // Fail-fast on the first encoding error: the position of later
  // trampolines depends on earlier ones, so a single bad branch would
  // cascade into incorrect layout. A single failure invalidates the whole
  // rewrite, so there is nothing useful to recover beyond it.
  uint64_t TrampOffset = TextSize;
  for (Trampoline &T : Trampolines) {
    uint64_t TP = TrampOffset;
    TrampOffset += T.Bytes.size();

    SmallVector<uint8_t> BrBack = LS.encodeSBranch(
        TP + T.Bytes.size() - MinInstSize, T.OriginalOffset + T.OriginalSize);
    if (BrBack.empty()) {
      log() << "hotswap: error: trampoline branch-back encoding failed at 0x"
            << utohexstr(T.OriginalOffset) << "\n";
      return false;
    }
    std::memcpy(T.Bytes.data() + T.Bytes.size() - MinInstSize, BrBack.data(),
                BrBack.size());

    SmallVector<uint8_t> BrFwd = LS.encodeSBranch(T.OriginalOffset, TP);
    if (BrFwd.empty()) {
      log() << "hotswap: error: trampoline branch-fwd encoding failed at 0x"
            << utohexstr(T.OriginalOffset) << "\n";
      return false;
    }
    std::memcpy(Text + T.OriginalOffset, BrFwd.data(), BrFwd.size());
    // Pad the tail of the replaced slot with cached s_nop bytes.
    for (uint32_t I = MinInstSize; I < T.OriginalSize; I += MinInstSize)
      std::memcpy(Text + T.OriginalOffset + I, LS.SNopBytes.data(),
                  MinInstSize);
  }
  return true;
}

/// Fix up DWARF sections of the grown ELF after trampolines have been
/// appended: adds trampoline symbols to the symbol table, shifts
/// .debug_line / .debug_ranges / .debug_info / .debug_frame addresses by
/// the total trampoline footprint, and reports per-section failures via
/// log(). Individual patchDebug* helpers are weak stubs here; concrete
/// implementations land in separate PRs.
static void patchDebugSections(WritableMemoryBuffer &ElfBuf,
                               ArrayRef<Trampoline> Trampolines,
                               const ElfView &Elf, size_t TrampTotal) {
  uint8_t *Data = reinterpret_cast<uint8_t *>(ElfBuf.getBufferStart());
  size_t Size = ElfBuf.getBufferSize();
  if (!addTrampolineSymbols(ElfBuf, Trampolines, Elf.textSize(),
                            Elf.textSectionIndex()))
    log() << "hotswap: error: addTrampolineSymbols failed\n";
  patchDebugRanges(Data, Size, Elf.textAddr(), Elf.textSize(), TrampTotal);
  patchDebugInfo(Data, Size, Elf.textAddr(), Elf.textSize(), TrampTotal);
  patchDebugFrame(Data, Size, Elf.textAddr(), Elf.textSize(), TrampTotal);
  if (!patchDebugLine(ElfBuf, Trampolines, Elf.textSize(), Elf.textAddr()))
    log() << "hotswap: error: patchDebugLine failed\n";
}

/// Re-open the grown ELF and cross-check that no scratch-patched site
/// reads a VGPR still live at the patch point: builds a fresh ElfView over
/// the output buffer, hands the new .text to verifyPatchCorrectness, and
/// logs a diagnostic if the verifier detects a potential conflict. Runs
/// only when the scratch patch pass produced at least one ScratchPatchInfo
/// record.
static void runScratchVerification(WritableMemoryBuffer &OutBuf,
                                   const LLVMState &LS,
                                   ArrayRef<ScratchPatchInfo> ScratchPatches,
                                   unsigned MaxVgprs) {
  // Build a fresh ElfView over the grown buffer to find the new .text.
  // WritableMemoryBuffer::getBufferStart() returns char *, so no const_cast
  // is needed on the way to ElfView::create's uint8_t * contract.
  uint8_t *Data = reinterpret_cast<uint8_t *>(OutBuf.getBufferStart());
  Expected<ElfView> ViewOrErr = ElfView::create(Data, OutBuf.getBufferSize());
  if (!ViewOrErr) {
    consumeError(ViewOrErr.takeError());
    return;
  }
  if (ViewOrErr->textSize() == 0)
    return;
  if (!verifyPatchCorrectness(ViewOrErr->textData(), ViewOrErr->textSize(), LS,
                              ScratchPatches, MaxVgprs))
    log() << "hotswap: error: post-patch verification detected possible "
          << "scratch conflicts\n";
}

// -- retargetCodeObjectB0A0 ---------------------------------------------------

amd_comgr_status_t retargetCodeObjectB0A0(const void *ElfData, size_t ElfSize,
                                          const TargetIdentifier &TargetIdent,
                                          std::unique_ptr<MemoryBuffer> &Out) {
  // Take a working copy so the input is preserved and we have a mutable
  // buffer to parse / patch.
  std::vector<uint8_t> Buf(static_cast<const uint8_t *>(ElfData),
                           static_cast<const uint8_t *>(ElfData) + ElfSize);

  Expected<ElfView> ViewOrErr = ElfView::create(Buf.data(), Buf.size());
  if (!ViewOrErr) {
    log() << "hotswap: error: retargetCodeObjectB0A0: input is not a "
          << "parseable ELF64 (" << toString(ViewOrErr.takeError()) << ").\n";
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }
  if (ViewOrErr->textSize() == 0) {
    log() << "hotswap: error: retargetCodeObjectB0A0: input ELF has empty "
          << ".text section; nothing to rewrite.\n";
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }
  ElfView &Elf = *ViewOrErr;

  LLVMState LS = initLLVM(TargetIdent);
  if (!LS.Valid) {
    log() << "hotswap: error: retargetCodeObjectB0A0: initLLVM failed "
          << "for CPU '" << TargetIdent.Processor << "'; aborting rewrite.\n";
    return AMD_COMGR_STATUS_ERROR;
  }

  RewriteConfig Config = makeGfx1250B0A0Config();

  uint8_t *Text = Elf.textData();
  std::vector<InternalDecodedInst> Decoded;
  if (!decodeTextSection(Text, Elf.textSize(), LS, Decoded)) {
    log() << "hotswap: error: retargetCodeObjectB0A0: decodeTextSection "
          << "failed on .text (" << Elf.textSize() << " bytes).\n";
    return AMD_COMGR_STATUS_ERROR;
  }

  std::vector<Trampoline> Deferred;
  std::vector<ScratchPatchInfo> ScratchPatches;
  uint32_t Count = applyGfx1250B0toA0Rules(
      Decoded, Text, Elf.textSize(), LS, Deferred, Elf, ScratchPatches, Config);

  log() << "hotswap: applied " << Count << " patches\n";

  std::unique_ptr<WritableMemoryBuffer> Result;
  if (!Deferred.empty()) {
    if (!fixupTrampolineBranches(Deferred, Text, Elf.textSize(), LS))
      log() << "hotswap: error: some trampolines could not be fixed up\n";

    Result = Elf.growWithTrampolines(Deferred, LS.SNopBytes);
    if (!Result) {
      log() << "hotswap: error: retargetCodeObjectB0A0: "
            << "ElfView::growWithTrampolines returned null with "
            << Deferred.size() << " trampolines queued.\n";
      return AMD_COMGR_STATUS_ERROR;
    }

    size_t TrampTotal = 0;
    for (const Trampoline &T : Deferred)
      TrampTotal += T.Bytes.size();
    patchDebugSections(*Result, Deferred, Elf, TrampTotal);
  } else {
    Result = WritableMemoryBuffer::getNewUninitMemBuffer(ElfSize);
    if (!Result) {
      log() << "hotswap: error: retargetCodeObjectB0A0: "
            << "getNewUninitMemBuffer(" << ElfSize
            << ") failed (out of memory) for the patched output copy.\n";
      return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
    }
    std::memcpy(Result->getBufferStart(), Buf.data(), ElfSize);
  }

  if (!ScratchPatches.empty())
    runScratchVerification(*Result, LS, ScratchPatches, Config.MaxVgprs);

  Out = std::move(Result);
  return AMD_COMGR_STATUS_SUCCESS;
}

} // namespace hotswap
} // namespace COMGR
