//===- comgr-hotswap-b0a0.cpp - GFX1250 B0-to-A0 patch dispatcher --------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Dispatcher for B0-to-A0 silicon stepping patches and the
/// retargetCodeObject orchestrator that drives the full pipeline:
/// decode -> patch -> trampoline growth -> DWARF update.
///
/// Patch passes are dispatched through HotswapPatchVTable. The membership
/// list lives in comgr-hotswap-patches.def; each entry corresponds to one
/// slot on the vtable and one register*Patch function in a sibling
/// comgr-hotswap-patch-*.cpp. installHotswapPatches() walks the .def to
/// bind every slot. The vtable is exposed through getHotswapPatchVTable(),
/// a Meyers singleton whose initializer eagerly runs installHotswapPatches
/// on its private storage; C++11 [stmt.dcl]/4 guarantees this happens
/// exactly once and is safe under concurrent first access, so the
/// dispatcher and the amd_comgr_hotswap_rewrite entry point can fetch the
/// fully-bound vtable with no explicit synchronization.
/// This replaces the prior LLVM_ATTRIBUTE_WEAK + `#if !defined(_MSC_VER)`
/// override pattern, which silently disabled hotswap on Windows because
/// PE/COFF does not honour weak the way ELF does
/// (issue ROCm/llvm-project#2479).
///
//===----------------------------------------------------------------------===//

#include "comgr-env.h"
#include "comgr-hotswap-internal.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <limits>
#include <mutex>

using namespace llvm;

namespace COMGR {
namespace hotswap {

// HotSwap rewrite profiling lives in comgr-hotswap-internal.h so the sibling
// comgr-hotswap-patch-*.cpp TUs can record into the same per-rewrite session.

// -- GFX1250 B0-to-A0 constants -----------------------------------------------
//
// All instruction encoding lives in LLVMState (s_branch opcode + pre-encoded
// s_nop bytes, populated at initLLVM time via the MC asm parser). This policy
// layer only carries ISA identifiers and register granularity -- no
// target-specific opcode bits should land here.

static constexpr unsigned Gfx1250MaxVgprs = 1024;
// GFX1250 wave32 VGPR ENCODING granularity is 16 (per
// AMDGPUBaseInfo::getVGPREncodingGranule with Feature1024AddressableVGPRs),
// not the 8 used by earlier GFX10/11 wave32. Used by ElfView's KD
// decode/encode helpers (getKernelVgprCount /
// updateKernelDescriptorVgprCount) to
// interpret COMPUTE_PGM_RSRC1.GRANULATED_WORKITEM_VGPR_COUNT.
// GFX12 wave32: 106 user-addressable SGPRs (s0-s105); s106-s107 are VCC.
static constexpr unsigned Gfx1250MaxSgprs = 106;
static constexpr unsigned Gfx1250VgprGranuleSize = 16;

/// Build the default RewriteConfig used for the GFX1250 B0-to-A0 rewrite:
/// fills in the identity source / target ISA (both gfx1250) and the
/// AMDGPU register granularity constants consumed by
/// ElfView::updateKernelDescriptorVgprCount. Instruction-encoding state is not
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
  Config.MaxSgprs = Gfx1250MaxSgprs;
  Config.VgprGranuleSize = Gfx1250VgprGranuleSize;
  return Config;
}

static bool appendCodeEndGuard(std::vector<Trampoline> &Growth,
                               uint64_t GuardBytes, const LLVMState &LS) {
  if (GuardBytes == 0)
    return true;

  SmallVector<uint8_t> CodeEnd = assembleSingleInst("s_code_end", LS);
  if (CodeEnd.empty()) {
    log() << "hotswap: error: failed to assemble s_code_end for trampoline "
          << "prefetch guard.\n";
    return false;
  }
  if (GuardBytes % CodeEnd.size() != 0) {
    log() << "hotswap: error: trampoline prefetch guard size " << GuardBytes
          << " is not a multiple of s_code_end size " << CodeEnd.size()
          << ".\n";
    return false;
  }

  Trampoline Guard;
  while (static_cast<uint64_t>(Guard.Bytes.size()) < GuardBytes)
    Guard.Bytes.append(CodeEnd.begin(), CodeEnd.end());
  Growth.push_back(std::move(Guard));
  return true;
}

static std::optional<uint32_t>
getMaxOriginalKernelInstPrefSize(const ElfView &Elf, const LLVMState &LS) {
  ArrayRef<KernelDescriptorInfo> Descriptors = Elf.kernelDescriptors();
  uint32_t MaxOriginalInstPrefLines = 0;
  for (const KernelDescriptorInfo &KD : Descriptors) {
    std::optional<uint32_t> OriginalInstPrefLines =
        Elf.getKernelDescriptorInstPrefSize(KD.KernelName, LS.Cpu);
    if (!OriginalInstPrefLines)
      return std::nullopt;
    MaxOriginalInstPrefLines =
        std::max(MaxOriginalInstPrefLines, *OriginalInstPrefLines);
  }
  return MaxOriginalInstPrefLines;
}

static bool
appendDeferredTrampolinePrefetchGuard(const ElfView &Elf, const LLVMState &LS,
                                      std::vector<Trampoline> &Growth) {
  // Deferred instruction-rewrite trampolines are reached from the original
  // kernel entries, so their trailing guard follows the original descriptor
  // prefetch size. Kernel-entry stubs clamp their own descriptor prefetch.
  std::optional<uint32_t> MaxOriginalInstPrefLines =
      getMaxOriginalKernelInstPrefSize(Elf, LS);
  if (!MaxOriginalInstPrefLines)
    return false;

  uint64_t GuardBytes = static_cast<uint64_t>(*MaxOriginalInstPrefLines) *
                        KernelEntryInstPrefUnitBytes;
  if (!appendCodeEndGuard(Growth, GuardBytes, LS))
    return false;

  log() << "hotswap: appended " << GuardBytes
        << " trampoline prefetch guard bytes\n";
  return true;
}

// -- Forward declarations for liveness/DWARF stubs ----------------------------
//
// These have weak default definitions below. The apply* patch families use
// HotswapPatchVTable dispatch; these lower-level helpers stay on weak stubs
// until a real implementation lands, at which point they should migrate to
// an explicit registration contract as well.

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

// -- HotswapPatchVTable plumbing ----------------------------------------------
//
// Patch-module forward declarations live in comgr-hotswap-internal.h
// (driven off the same comgr-hotswap-patches.def), so libamd_comgr and
// the unit tests share one prototype source. Here we supply the
// singleton accessor and the installer that walks the .def to invoke
// each register*Patch. A .def entry without a matching register*Patch
// definition produces a link error at libamd_comgr link time.
//
// installHotswapPatches() is exposed in the header so unit tests can
// bind a local HotswapPatchVTable for fixture-style coverage. Production
// code never calls it directly: getHotswapPatchVTable()'s initializer
// invokes it eagerly on the singleton's private storage, which the C++11
// magic-static rule guarantees runs exactly once even under concurrent
// first access. That removes both the explicit std::call_once at the
// retargetCodeObject entry point and any inter-TU static-init order
// dependency on the patch modules.

void installHotswapPatches(HotswapPatchVTable &VT) {
#define HOTSWAP_PATCH(Name) register##Name##Patch(VT);
#include "comgr-hotswap-patches.def"
#undef HOTSWAP_PATCH
}

HotswapPatchVTable &getHotswapPatchVTable() {
  static HotswapPatchVTable VT = [] {
    HotswapPatchVTable Tmp;
    installHotswapPatches(Tmp);
    return Tmp;
  }();
  return VT;
}

// -- Weak-symbol liveness stubs -----------------------------------------------
//
// Conservative defaults: all VGPRs reported live. VgprAllocator will
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

static void appendNopSledIfLarge(std::vector<NopSled> &Sleds, uint64_t Start,
                                 uint64_t End, uint64_t FunctionStart,
                                 uint64_t FunctionEnd) {
  if (End - Start >= MinNopSledSize)
    Sleds.push_back({Start, End, Start, FunctionStart, FunctionEnd});
}

static void appendNopSledIfLarge(std::vector<NopSled> &Sleds, uint64_t Start,
                                 uint64_t End,
                                 const ElfView::FunctionTextRange &Range) {
  appendNopSledIfLarge(Sleds, Start, End, Range.Begin, Range.End);
}

/// Scan \p Decoded for runs of consecutive `s_nop` instructions at least
/// MinNopSledSize bytes long and return the resulting NopSled list. Each sled
/// records its owning function range so emitReplacementCode can only borrow
/// padding from the same kernel as the instruction being patched. NOPs outside
/// any sized function symbol are ignored.
static std::vector<NopSled>
buildNopSledMap(ArrayRef<InternalDecodedInst> Decoded, const LLVMState &LS,
                const ElfView &Elf) {
  std::vector<NopSled> Sleds;
  bool HasActiveRange = false;
  ElfView::FunctionTextRange ActiveRange;
  uint64_t Start = 0;
  uint64_t End = 0;

  for (const InternalDecodedInst &DI : Decoded) {
    if (DI.Inst.getOpcode() != LS.SNopOpcode) {
      if (HasActiveRange)
        appendNopSledIfLarge(Sleds, Start, End, ActiveRange);
      HasActiveRange = false;
      continue;
    }

    std::optional<ElfView::FunctionTextRange> Range =
        Elf.findFunctionTextRangeAtOffset(DI.Offset);
    if (!Range || DI.Size > Range->End - DI.Offset) {
      if (HasActiveRange)
        appendNopSledIfLarge(Sleds, Start, End, ActiveRange);
      HasActiveRange = false;
      continue;
    }

    if (!HasActiveRange || ActiveRange.Begin != Range->Begin ||
        ActiveRange.End != Range->End || DI.Offset != End) {
      if (HasActiveRange)
        appendNopSledIfLarge(Sleds, Start, End, ActiveRange);
      ActiveRange = *Range;
      HasActiveRange = true;
      Start = DI.Offset;
    }
    End = DI.Offset + DI.Size;
  }

  if (HasActiveRange)
    appendNopSledIfLarge(Sleds, Start, End, ActiveRange);
  return Sleds;
}

/// A direct branch/call target into a NOP run makes that offset and every
/// following byte in the run reachable by fallthrough, so only the prefix
/// before the first target remains available as scratch padding.
static void
truncateNopSledsAtDirectTargets(std::vector<NopSled> &Sleds,
                                const DenseSet<uint64_t> &DirectBranchTargets) {
  if (DirectBranchTargets.empty() || Sleds.empty())
    return;

  std::vector<NopSled> Filtered;
  Filtered.reserve(Sleds.size());
  uint64_t Truncated = 0;
  for (const NopSled &Sled : Sleds) {
    uint64_t End = Sled.End;
    for (uint64_t Target : DirectBranchTargets)
      if (Target >= Sled.Start && Target < End)
        End = Target;
    if (End != Sled.End)
      ++Truncated;
    appendNopSledIfLarge(Filtered, Sled.Start, End, Sled.FunctionStart,
                         Sled.FunctionEnd);
  }
  if (Truncated != 0)
    log() << "hotswap: protected " << Truncated
          << " NOP sled(s) containing direct branch/call target(s)\n";
  Sleds = std::move(Filtered);
}

// -- Sled-or-trampoline code emission -----------------------------------------

/// Emit the replacement code for the instruction at [\p InstOffset,
/// \p InstOffset + \p InstSize) into a nearby NOP sled: writes \p Replacement
/// into the sled, appends a branch-back to the next instruction after the
/// original site, overwrites the original site with a branch-forward to the
/// sled, and pads the leftover bytes of the original slot with cached s_nop
/// bytes. Advances \c Sled.WritePos by the amount consumed. Returns false if
/// either branch encoding fails. Branches are encoded before any bytes are
/// written so a failure leaves \c Ctx.Text and \c Sled.WritePos unchanged.
[[nodiscard]] bool emitToNopSled(PatchContext &Ctx, NopSled &Sled,
                                 uint64_t InstOffset, uint32_t InstSize,
                                 ArrayRef<uint8_t> Replacement) {
  const LLVMState &LS = Ctx.LS;
  SmallVector<uint8_t> BrBack = LS.encodeSBranch(
      Sled.WritePos + Replacement.size(), InstOffset + InstSize);
  if (BrBack.empty()) {
    log() << "hotswap: error: emitToNopSled: encodeSBranch for branch-back "
          << "at sled offset 0x"
          << utohexstr(Sled.WritePos + Replacement.size()) << " -> 0x"
          << utohexstr(InstOffset + InstSize) << " failed.\n";
    return false;
  }

  SmallVector<uint8_t> BrFwd = LS.encodeSBranch(InstOffset, Sled.WritePos);
  if (BrFwd.empty()) {
    log() << "hotswap: error: emitToNopSled: encodeSBranch for branch-fwd "
          << "at original offset 0x" << utohexstr(InstOffset) << " -> sled 0x"
          << utohexstr(Sled.WritePos) << " failed.\n";
    return false;
  }

  std::memcpy(Ctx.Text + Sled.WritePos, Replacement.data(), Replacement.size());
  std::memcpy(Ctx.Text + Sled.WritePos + Replacement.size(), BrBack.data(),
              BrBack.size());
  std::memcpy(Ctx.Text + InstOffset, BrFwd.data(), BrFwd.size());

  // Pad the tail of the replaced instruction slot with cached s_nop bytes
  // (pre-encoded in LLVMState at initLLVM() time).
  for (uint32_t I = MinInstSize; I < InstSize; I += MinInstSize)
    std::memcpy(Ctx.Text + InstOffset + I, LS.SNopBytes.data(), MinInstSize);

  Sled.WritePos += Replacement.size() + MinInstSize;
  // Count-only row: patch placed in-line via a nearby NOP sled, no trampoline.
  Ctx.Profile.count(HotswapMetric::JumpNopSled);
  return true;
}

std::optional<SmallVector<uint8_t>> encodeSetPCLongBranch(const LLVMState &LS,
                                                          uint64_t FromOffset,
                                                          uint64_t TargetOffset,
                                                          unsigned SgprBase) {
  if ((SgprBase & 1u) != 0) {
    log() << "hotswap: error: set-PC long branch requires an aligned "
             "SGPR pair, got s"
          << SgprBase << "\n";
    return std::nullopt;
  }

  const std::string Pair = "s[" + std::to_string(SgprBase) + ":" +
                           std::to_string(SgprBase + 1) + "]";
  SmallVector<uint8_t> GetPc = assembleSingleInst("s_get_pc_i64 " + Pair, LS);
  if (GetPc.empty())
    return std::nullopt;
  std::optional<uint64_t> PcBase =
      checkedAddUint64(FromOffset, GetPc.size(), "set-PC long branch PC base");
  if (!PcBase)
    return std::nullopt;
  uint64_t Delta = TargetOffset - *PcBase;
  // AMDGPU/SOPInstructions.td defines S_ADD_U64 as an SOP2_64 outside the
  // Defs = [SCC] scope and maps its gfx12 encoding to s_add_nc_u64. It can
  // therefore add the complete PC displacement without saving or clobbering
  // SCC.
  SmallVector<std::string, 3> AsmLines;
  AsmLines.push_back("s_get_pc_i64 " + Pair);
  AsmLines.push_back("s_add_nc_u64 " + Pair + ", " + Pair + ", 0x" +
                     utohexstr(Delta));
  AsmLines.push_back("s_set_pc_i64 " + Pair);
  SmallVector<uint8_t> Bytes = assembleInstructions(joinAsmLines(AsmLines), LS);
  if (Bytes.empty() || Bytes.size() > SetPcReturnReserveBytes) {
    log() << "hotswap: error: failed to assemble SCC-neutral set-PC branch via "
          << Pair << "\n";
    return std::nullopt;
  }
  return Bytes;
}

Expected<std::optional<EncodedSetPcGateway>>
findNearestSetPcGateway(std::vector<NopSled> &Gateways, const LLVMState &LS,
                        uint64_t FromOffset, uint64_t TargetOffset,
                        unsigned SgprBase) {
  NopSled *Best = nullptr;
  SmallVector<uint8_t> BestBytes;
  uint64_t BestDistance = std::numeric_limits<uint64_t>::max();
  for (NopSled &Sled : Gateways) {
    if (FromOffset < Sled.FunctionStart || FromOffset >= Sled.FunctionEnd)
      continue;
    uint64_t UsableEnd = std::min(Sled.End, Sled.FunctionEnd);
    if (Sled.WritePos > UsableEnd)
      continue;
    uint64_t Distance = Sled.WritePos > FromOffset ? Sled.WritePos - FromOffset
                                                   : FromOffset - Sled.WritePos;
    if (Distance >= MaxSledDistance || Distance >= BestDistance ||
        LS.encodeSBranch(FromOffset, Sled.WritePos).empty())
      continue;
    std::optional<SmallVector<uint8_t>> Bytes =
        encodeSetPCLongBranch(LS, Sled.WritePos, TargetOffset, SgprBase);
    if (!Bytes)
      return createStringError(
          Twine("failed to encode set-PC gateway at candidate offset 0x") +
          utohexstr(Sled.WritePos));
    if (Bytes->size() > UsableEnd - Sled.WritePos)
      continue;
    Best = &Sled;
    BestBytes = std::move(*Bytes);
    BestDistance = Distance;
  }
  if (!Best)
    return std::nullopt;
  return std::optional<EncodedSetPcGateway>(
      EncodedSetPcGateway{Best, std::move(BestBytes)});
}

static std::optional<unsigned> numberedSgprIndex(const MCRegisterInfo &MRI,
                                                 MCRegister Reg) {
  // TODO(https://github.com/ROCm/llvm-project/issues/3350): Replace this
  // register-name fallback with a public AMDGPU MC hardware-index helper.
  if (!Reg.isValid())
    return std::nullopt;
  StringRef Name(MRI.getName(Reg));
  if (!Name.consume_front("SGPR") || Name.empty() || Name.contains('_'))
    return std::nullopt;
  unsigned Index = 0;
  if (Name.getAsInteger(10, Index))
    return std::nullopt;
  return Index;
}

static bool updateNumberedSgprHighWatermark(const MCRegisterInfo &MRI,
                                            MCRegister Reg, unsigned MaxSgprs,
                                            unsigned &HighWatermark,
                                            StringRef Context) {
  SmallVector<MCRegister, 8> Candidates;
  Candidates.push_back(Reg);
  for (MCPhysReg Sub : MRI.subregs(Reg))
    Candidates.push_back(MCRegister(Sub));

  for (MCRegister Candidate : Candidates) {
    std::optional<unsigned> Index = numberedSgprIndex(MRI, Candidate);
    if (!Index)
      continue;
    if (*Index >= MaxSgprs) {
      log() << "hotswap: error: " << Context << ": numbered SGPR s" << *Index
            << " exceeds the addressable limit s" << (MaxSgprs - 1) << "\n";
      return false;
    }
    HighWatermark = std::max(HighWatermark, *Index + 1);
  }
  return true;
}

static bool isVccRegister(const LLVMState &LS, MCRegister Reg) {
  return Reg.isValid() && LS.VCCRegister.isValid() &&
         LS.MRI->regsOverlap(Reg, LS.VCCRegister);
}

static bool instructionUsesVcc(const LLVMState &LS,
                               const InternalDecodedInst &DI) {
  for (const MCOperand &Op : DI.Inst)
    if (Op.isReg() && Op.getReg() && isVccRegister(LS, MCRegister(Op.getReg())))
      return true;

  const MCInstrDesc &Desc = LS.MCII->get(DI.Inst.getOpcode());
  for (MCPhysReg Reg : Desc.implicit_uses())
    if (isVccRegister(LS, MCRegister(Reg)))
      return true;
  for (MCPhysReg Reg : Desc.implicit_defs())
    if (isVccRegister(LS, MCRegister(Reg)))
      return true;
  return false;
}

static SafeSgprUsageSummary
summarizeSafeSgprUsage(PatchContext &Ctx,
                       ArrayRef<InternalDecodedInst> Instructions,
                       StringRef Context) {
  SafeSgprUsageSummary Summary;
  for (const InternalDecodedInst &DI : Instructions) {
    Summary.UsesVcc |= instructionUsesVcc(Ctx.LS, DI);
    Summary.HasCall |= Ctx.LS.MIA && Ctx.LS.MIA->isCall(DI.Inst);
    for (const MCOperand &Op : DI.Inst) {
      if (!Op.isReg() || !Op.getReg())
        continue;
      if (!updateNumberedSgprHighWatermark(*Ctx.LS.MRI, MCRegister(Op.getReg()),
                                           Ctx.Config.MaxSgprs,
                                           Summary.HighWatermark, Context)) {
        Summary.Valid = false;
        return Summary;
      }
    }

    const MCInstrDesc &Desc = Ctx.LS.MCII->get(DI.Inst.getOpcode());
    for (MCPhysReg Reg : Desc.implicit_uses())
      if (!updateNumberedSgprHighWatermark(*Ctx.LS.MRI, MCRegister(Reg),
                                           Ctx.Config.MaxSgprs,
                                           Summary.HighWatermark, Context)) {
        Summary.Valid = false;
        return Summary;
      }
    for (MCPhysReg Reg : Desc.implicit_defs())
      if (!updateNumberedSgprHighWatermark(*Ctx.LS.MRI, MCRegister(Reg),
                                           Ctx.Config.MaxSgprs,
                                           Summary.HighWatermark, Context)) {
        Summary.Valid = false;
        return Summary;
      }
  }
  return Summary;
}

std::optional<SafeSgprScratchBlock>
findSafeSgprScratchBlock(PatchContext &Ctx, uint64_t TextOffset, unsigned Count,
                         unsigned Alignment, StringRef Context) {
  if (Count == 0 || Alignment == 0 || (Alignment & (Alignment - 1)) != 0) {
    log() << "hotswap: error: " << Context
          << ": invalid global SGPR block request (count=" << Count
          << ", alignment=" << Alignment << ")\n";
    return std::nullopt;
  }

  std::optional<ElfView::FunctionTextRange> FunctionRange =
      Ctx.Elf.findFunctionTextRangeAtOffset(TextOffset);
  std::string Owner =
      Ctx.Elf.findKernelAtAddress(TextOffset + Ctx.Elf.textAddr());
  bool ScanWholeObject = Owner.empty() || !FunctionRange;
  SafeSgprUsageSummary *Usage = nullptr;
  if (!ScanWholeObject) {
    using FunctionKey = std::pair<uint64_t, uint64_t>;
    FunctionKey Key{FunctionRange->Begin, FunctionRange->End};
    DenseMap<FunctionKey, SafeSgprUsageSummary>::iterator Cached =
        Ctx.FunctionSgprUsage.find(Key);
    if (Cached == Ctx.FunctionSgprUsage.end()) {
      std::vector<InternalDecodedInst>::const_iterator Begin = std::lower_bound(
          Ctx.Decoded.cbegin(), Ctx.Decoded.cend(), FunctionRange->Begin,
          [](const InternalDecodedInst &DI, uint64_t Offset) {
            return DI.Offset < Offset;
          });
      std::vector<InternalDecodedInst>::const_iterator End =
          std::lower_bound(Begin, Ctx.Decoded.cend(), FunctionRange->End,
                           [](const InternalDecodedInst &DI, uint64_t Offset) {
                             return DI.Offset < Offset;
                           });
      size_t BeginIndex = Begin - Ctx.Decoded.cbegin();
      size_t InstructionCount = End - Begin;
      SafeSgprUsageSummary Summary =
          summarizeSafeSgprUsage(Ctx,
                                 ArrayRef<InternalDecodedInst>(Ctx.Decoded)
                                     .slice(BeginIndex, InstructionCount),
                                 Context);
      Cached = Ctx.FunctionSgprUsage.try_emplace(Key, Summary).first;
    }
    Usage = &Cached->second;
    ScanWholeObject = Usage->HasCall;
  }

  if (ScanWholeObject) {
    if (!Ctx.WholeObjectSgprUsage)
      Ctx.WholeObjectSgprUsage = summarizeSafeSgprUsage(
          Ctx, ArrayRef<InternalDecodedInst>(Ctx.Decoded), Context);
    Usage = &*Ctx.WholeObjectSgprUsage;
  }
  if (!Usage || !Usage->Valid) {
    log() << "hotswap: error: " << Context
          << ": cached SGPR usage analysis failed\n";
    return std::nullopt;
  }

  bool UsesVcc = Usage->UsesVcc;
  unsigned HighWatermark = Usage->HighWatermark;

  constexpr unsigned VccSgprs = 2;
  if (!Owner.empty()) {
    std::optional<unsigned> Declared = Ctx.Elf.getKernelSgprCount(Owner);
    if (!Declared) {
      log() << "hotswap: error: " << Context
            << ": failed to read SGPR count for kernel " << Owner << "\n";
      return std::nullopt;
    }
    if (UsesVcc && *Declared < VccSgprs) {
      log() << "hotswap: error: " << Context << ": VCC-using kernel " << Owner
            << " has invalid SGPR count " << *Declared << "\n";
      return std::nullopt;
    }
    unsigned DeclaredNumbered = *Declared - (UsesVcc ? VccSgprs : 0);
    HighWatermark = std::max(HighWatermark, DeclaredNumbered);
  } else {
    // A device function can be reached from kernels with different declared
    // register footprints. Without a complete call graph, keep the block above
    // every declaration and charge every kernel in the commit step.
    for (const KernelDescriptorInfo &KD : Ctx.Elf.kernelDescriptors()) {
      std::optional<unsigned> Declared =
          Ctx.Elf.getKernelSgprCount(KD.KernelName);
      if (!Declared) {
        log() << "hotswap: error: " << Context
              << ": failed to read SGPR count for kernel " << KD.KernelName
              << "\n";
        return std::nullopt;
      }
      HighWatermark = std::max(HighWatermark, *Declared);
    }
  }

  if (HighWatermark > std::numeric_limits<unsigned>::max() - (Alignment - 1)) {
    log() << "hotswap: error: " << Context
          << ": SGPR alignment calculation overflows unsigned\n";
    return std::nullopt;
  }
  unsigned Base = (HighWatermark + Alignment - 1) & ~(Alignment - 1);
  if (Base > Ctx.Config.MaxSgprs || Count > Ctx.Config.MaxSgprs - Base) {
    log() << "hotswap: error: " << Context << ": no aligned block of " << Count
          << " safe SGPRs fits below s" << Ctx.Config.MaxSgprs << "\n";
    return std::nullopt;
  }
  return SafeSgprScratchBlock{Base, Count};
}

bool commitSafeSgprScratchBlock(PatchContext &Ctx, uint64_t TextOffset,
                                const SafeSgprScratchBlock &Block,
                                StringRef Context) {
  ArrayRef<KernelDescriptorInfo> Descriptors = Ctx.Elf.kernelDescriptors();
  if (Descriptors.empty()) {
    log() << "hotswap: error: " << Context
          << ": code object has no kernel descriptors to charge for scratch "
             "SGPRs\n";
    return false;
  }

  std::string Owner =
      Ctx.Elf.findKernelAtAddress(TextOffset + Ctx.Elf.textAddr());
  bool ChargedOwner = false;

  // llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp::getNumExtraSGPRs returns
  // two non-numbered VCC SGPRs on GFX1250. Always include them in the metadata
  // requirement. This may conservatively overstate a kernel that does not use
  // VCC, but never mistakes VCC for numbered s0-s105 registers.
  constexpr unsigned VccSgprs = 2;
  unsigned RequiredSgprs = Block.Base + Block.Count + VccSgprs;
  for (const KernelDescriptorInfo &KD : Descriptors) {
    if (!Owner.empty() && KD.KernelName != Owner)
      continue;
    ChargedOwner = true;

    std::optional<unsigned> Current = Ctx.Elf.getKernelSgprCount(KD.KernelName);
    if (!Current) {
      log() << "hotswap: error: " << Context
            << ": failed to read SGPR count for kernel " << KD.KernelName
            << "\n";
      return false;
    }
    if (*Current >= RequiredSgprs)
      continue;
    KernelPatchStats &Stats = Ctx.KernelStats[KD.KernelName];
    Stats.ExtraSgprs = std::max(Stats.ExtraSgprs, RequiredSgprs - *Current);
  }

  if (!ChargedOwner) {
    log() << "hotswap: error: " << Context << ": kernel '" << Owner
          << "' has no descriptor\n";
    return false;
  }
  return true;
}

static std::optional<SafeSgprScratchBlock>
reserveSafeFarReturn(PatchContext &Ctx, uint64_t InstOffset) {
  std::optional<SafeSgprScratchBlock> Scratch = findSafeSgprScratchBlock(
      Ctx, InstOffset, /*Count=*/2, /*Alignment=*/2, "safe far return");
  if (!Scratch)
    return std::nullopt;
  if (!commitSafeSgprScratchBlock(Ctx, InstOffset, *Scratch, "safe far return"))
    return std::nullopt;
  return Scratch;
}

bool isSBranchReachable(uint64_t From, uint64_t To) {
  std::optional<uint64_t> PcBase =
      checkedAddUint64(From, MinInstSize, "short branch PC base");
  if (!PcBase)
    return false;
  uint64_t Delta = To >= *PcBase ? To - *PcBase : *PcBase - To;
  if (Delta % MinInstSize != 0)
    return false;
  uint64_t MaxDelta =
      To >= *PcBase ? static_cast<uint64_t>(BranchOffsetMax) * MinInstSize
                    : static_cast<uint64_t>(-BranchOffsetMin) * MinInstSize;
  return Delta <= MaxDelta;
}

/// Queue a deferred trampoline for [\p InstOffset, +\p InstSize) with
/// \p Replacement as its body; fixupTrampolineBranches fills in the edges once
/// the pool layout is known. A site beyond s_branch reach of the appended pool
/// uses an SCC-neutral get-PC/add/set-PC sequence on the backward edge.
/// Adjacent far sites are coalesced after patching to reduce gateway pressure.
/// Every far source edge then uses a short branch to nearby safe NOP padding;
/// that gateway uses the gfx12 SGPR-backed set-PC sequence. No source or return
/// edge executes gfx1250's broken s_add_pc_i64 instruction.
[[nodiscard]] bool emitToTrampoline(PatchContext &Ctx, uint64_t InstOffset,
                                    uint32_t InstSize,
                                    ArrayRef<uint8_t> Replacement) {
  // This trampoline lands at the appended pool base and after every trampoline
  // already queued -- later ones are appended behind it and cannot shift it,
  // and fixupTrampolineBranches walks the same list in the same order -- so its
  // final pool offset (relative to .text) is known exactly now.
  std::optional<uint64_t> PoolStart = checkedAddUint64(
      Ctx.PoolBaseOffset, Ctx.QueuedTrampolineBytes, "trampoline pool layout");
  if (!PoolStart)
    return false;

  // An s_branch encodes To - From as a signed simm16 dword field, in range iff
  // (To - From - MinInstSize) / MinInstSize fits [BranchOffsetMin,
  // BranchOffsetMax] (see LLVMState::encodeSBranch). Test both edges with the
  // short branch-back slot; the branch-back (pool tail -> site) is the farther
  // of the two. Go long only when a short branch cannot reach.
  std::optional<uint64_t> ShortBackFrom = checkedAddUint64(
      *PoolStart, Replacement.size(), "short trampoline return slot");
  std::optional<uint64_t> ReturnTo =
      checkedAddUint64(InstOffset, InstSize, "trampoline return target");
  if (!ShortBackFrom || !ReturnTo)
    return false;
  const bool Far = !(isSBranchReachable(InstOffset, *PoolStart) &&
                     isSBranchReachable(*ShortBackFrom, *ReturnTo));

  uint64_t ReturnReserve = Far ? SetPcReturnReserveBytes : MinInstSize;
  std::optional<uint64_t> TrampolineSize = checkedAddUint64(
      Replacement.size(), ReturnReserve, "queued trampoline size");
  if (!TrampolineSize)
    return false;
  std::optional<uint64_t> QueuedBytes =
      checkedAddUint64(Ctx.QueuedTrampolineBytes, *TrampolineSize,
                       "queued trampoline byte count");
  if (!QueuedBytes)
    return false;

  Trampoline T;
  T.OriginalOffset = InstOffset;
  T.OriginalSize = InstSize;
  T.Bytes.insert(T.Bytes.end(), Replacement.begin(), Replacement.end());
  if (std::optional<ElfView::FunctionTextRange> Range =
          Ctx.Elf.findFunctionTextRangeAtOffset(InstOffset)) {
    T.HasFunctionRange = true;
    T.FunctionStart = Range->Begin;
    T.FunctionEnd = Range->End;
  }

  if (Far) {
    // Every decline of a valid far site increments jump:declined_far (a
    // count-only row) so the metric reflects all placement failures, including
    // resource pressure, not just the size guard.
    auto declineFar = [&](const Twine &Reason) {
      Ctx.Profile.count(HotswapMetric::JumpDeclined);
      log() << "hotswap: far trampoline site 0x" << utohexstr(InstOffset)
            << " declined: " << Reason << "\n";
      return false;
    };
    if (InstSize < MinInstSize)
      return declineFar(Twine(InstSize) + " B, smaller than " +
                        Twine(MinInstSize) + " B forward branch");
    std::optional<SafeSgprScratchBlock> Scratch =
        reserveSafeFarReturn(Ctx, InstOffset);
    if (!Scratch)
      return declineFar("no safe SGPR triple for set-PC return");
    T.Bytes.insert(T.Bytes.end(), SetPcReturnReserveBytes, uint8_t{0});
    T.Long = true;
    T.UsesSetPCBack = true;
    T.LongBranchSgprBase = Scratch->Base;
    Ctx.Profile.count(HotswapMetric::JumpLong);
    Ctx.OutTrampolines.emplace_back(std::move(T));
    Ctx.QueuedTrampolineBytes = *QueuedBytes;
    return true;
  }
  {
    Ctx.Profile.count(HotswapMetric::JumpShort);
    // Reserve the short branch-back slot; fixupTrampolineBranches fills it in.
    T.Bytes.insert(T.Bytes.end(), MinInstSize, uint8_t{0});
  }
  Ctx.OutTrampolines.emplace_back(std::move(T));
  Ctx.QueuedTrampolineBytes = *QueuedBytes;
  return true;
}

std::optional<uint64_t>
evaluateDirectControlFlowTarget(const InternalDecodedInst &DI,
                                const LLVMState &LS) {
  uint64_t Target = 0;
  if (LS.MIA->evaluateBranch(DI.Inst, DI.Offset, DI.Size, Target))
    return Target;

  // TODO(https://github.com/ROCm/llvm-project/issues/3351): Remove this
  // fallback when AMDGPUMCInstrAnalysis::evaluateBranch locates the descriptor
  // operand marked MCOI::OPERAND_PCREL. Its current operand-zero restriction
  // is in llvm/lib/Target/AMDGPU/MCTargetDesc/AMDGPUMCTargetDesc.cpp.
  // GFX1250 s_call_i64 instead has its destination SGPR pair in slot zero and
  // its simm16 dword displacement in slot one; the operand layout and width
  // are pinned by llvm/test/MC/AMDGPU/gfx1250_asm_sopk.s.
  if (DI.Inst.getOpcode() != LS.SCallI64Opcode ||
      DI.Inst.getNumOperands() == 0 ||
      !DI.Inst.getOperand(DI.Inst.getNumOperands() - 1).isImm())
    return std::nullopt;

  uint64_t Encoded =
      static_cast<uint64_t>(
          DI.Inst.getOperand(DI.Inst.getNumOperands() - 1).getImm()) &
      0xFFFFu;
  int64_t DwordDelta = SignExtend64<16>(Encoded);
  std::optional<uint64_t> PcBase = checkedAddUint64(
      DI.Offset, DI.Size, "direct control-flow target PC base");
  if (!PcBase)
    return std::nullopt;
  if (DwordDelta >= 0)
    return checkedAddUint64(*PcBase,
                            static_cast<uint64_t>(DwordDelta) * MinInstSize,
                            "direct control-flow target");
  return checkedSubUint64(*PcBase,
                          static_cast<uint64_t>(-DwordDelta) * MinInstSize,
                          "direct control-flow target");
}

static bool definesOverlappingRegister(const InternalDecodedInst &DI,
                                       const LLVMState &LS,
                                       MCRegister Register) {
  const MCInstrDesc &Desc = LS.MCII->get(DI.Inst.getOpcode());
  unsigned DefCount = std::min(Desc.getNumDefs(), DI.Inst.getNumOperands());
  for (unsigned I = 0; I != DefCount; ++I) {
    const MCOperand &Operand = DI.Inst.getOperand(I);
    if (Operand.isReg() && Operand.getReg() &&
        LS.MRI->regsOverlap(MCRegister(Operand.getReg()), Register))
      return true;
  }
  if (Desc.variadicOpsAreDefs()) {
    unsigned VariadicBegin =
        std::min(Desc.getNumOperands(), DI.Inst.getNumOperands());
    for (unsigned I = VariadicBegin; I != DI.Inst.getNumOperands(); ++I) {
      const MCOperand &Operand = DI.Inst.getOperand(I);
      if (Operand.isReg() && Operand.getReg() &&
          LS.MRI->regsOverlap(MCRegister(Operand.getReg()), Register))
        return true;
    }
  }
  for (MCPhysReg ImplicitDef : Desc.implicit_defs())
    if (LS.MRI->regsOverlap(MCRegister(ImplicitDef), Register))
      return true;
  return false;
}

static bool isControlFlowBoundary(const InternalDecodedInst &DI,
                                  const LLVMState &LS) {
  return DI.Inst.getOpcode() == LS.SEndPgmOpcode ||
         DI.Inst.getOpcode() == LS.SEndPgmSavedOpcode ||
         LS.MIA->isBranch(DI.Inst) || LS.MIA->isCall(DI.Inst) ||
         LS.MIA->isReturn(DI.Inst) || LS.MIA->isIndirectBranch(DI.Inst) ||
         LS.MIA->isBarrier(DI.Inst);
}

struct DeclaredTextEntryInfo {
  SmallVector<uint64_t, 16> Entries;
  SmallVector<uint64_t, 16> ExternalEntries;
};

static std::optional<DeclaredTextEntryInfo>
collectDeclaredTextEntries(const ElfView &Elf) {
  std::optional<uint64_t> TextEnd =
      checkedAddUint64(Elf.textAddr(), Elf.textSize(), "declared text end");
  if (!TextEnd)
    return std::nullopt;

  DeclaredTextEntryInfo Info;
  for (const ElfView::FunctionTextRange &Range : Elf.functionTextRanges())
    if (Range.Begin >= Elf.textAddr() && Range.Begin < *TextEnd) {
      uint64_t Entry = Range.Begin - Elf.textAddr();
      Info.Entries.push_back(Entry);
      if (Range.Symbol && Range.Symbol->getBinding() != ELF::STB_LOCAL)
        Info.ExternalEntries.push_back(Entry);
    }

  for (const KernelDescriptorInfo &Descriptor : Elf.kernelDescriptors()) {
    std::optional<uint64_t> EntryAddress;
    if (Descriptor.EntryOffset >= 0) {
      EntryAddress = checkedAddUint64(
          Descriptor.VAddr, static_cast<uint64_t>(Descriptor.EntryOffset),
          "kernel descriptor entry address");
    } else {
      uint64_t Magnitude =
          Descriptor.EntryOffset == std::numeric_limits<int64_t>::min()
              ? uint64_t{1} << 63
              : static_cast<uint64_t>(-Descriptor.EntryOffset);
      EntryAddress = checkedSubUint64(Descriptor.VAddr, Magnitude,
                                      "kernel descriptor entry address");
    }
    if (!EntryAddress)
      return std::nullopt;
    if (*EntryAddress >= Elf.textAddr() && *EntryAddress < *TextEnd) {
      uint64_t Entry = *EntryAddress - Elf.textAddr();
      Info.Entries.push_back(Entry);
      Info.ExternalEntries.push_back(Entry);
    }
  }
  return Info;
}

struct PcMaterializedCallInfo {
  uint64_t Target = 0;
  uint64_t SequenceStart = 0;
  uint64_t SequenceEnd = 0;
  MCRegister ReturnRegister;
};

/// Resolve the compiler-emitted PC materialization used by the production
/// reproducer:
///
///   s_get_pc_i64 Target
///   ...                         // no Target definition or control flow
///   s_add_nc_u64 Target, Target, Immediate
///   ...                         // no Target definition or control flow
///   s_swap_pc_i64 Return, Target
///
/// The opcode and operand layout are defined by SOPInstructions.td and pinned
/// by llvm/test/MC/AMDGPU/gfx1250_asm_salu_lit64.s. Stop at the first
/// overlapping definition or control-flow boundary, so any variation remains
/// unresolved and follows the existing fail-closed policy.
static std::optional<PcMaterializedCallInfo>
matchPcMaterializedCall(ArrayRef<InternalDecodedInst> Decoded, size_t CallIndex,
                        const LLVMState &LS, uint64_t TextAddr) {
  const InternalDecodedInst &Call = Decoded[CallIndex];
  if (!Call.DecodeSucceeded || Call.Inst.getOpcode() != LS.SSwapPcI64Opcode ||
      Call.Inst.getNumOperands() < 2 || !Call.Inst.getOperand(0).isReg() ||
      !Call.Inst.getOperand(0).getReg())
    return std::nullopt;
  const MCOperand &TargetOperand =
      Call.Inst.getOperand(Call.Inst.getNumOperands() - 1);
  if (!TargetOperand.isReg() || !TargetOperand.getReg())
    return std::nullopt;
  MCRegister TargetRegister(TargetOperand.getReg());

  std::optional<size_t> AddIndex;
  int64_t AddImmediate = 0;
  for (size_t I = CallIndex; I != 0;) {
    --I;
    const InternalDecodedInst &Candidate = Decoded[I];
    if (!Candidate.DecodeSucceeded || isControlFlowBoundary(Candidate, LS))
      return std::nullopt;
    if (!definesOverlappingRegister(Candidate, LS, TargetRegister))
      continue;
    if (Candidate.Inst.getOpcode() != LS.SAddNcU64Opcode ||
        Candidate.Inst.getNumOperands() != 3 ||
        !Candidate.Inst.getOperand(0).isReg() ||
        Candidate.Inst.getOperand(0).getReg() != TargetRegister ||
        !Candidate.Inst.getOperand(1).isReg() ||
        Candidate.Inst.getOperand(1).getReg() != TargetRegister ||
        !Candidate.Inst.getOperand(2).isImm())
      return std::nullopt;
    AddIndex = I;
    AddImmediate = Candidate.Inst.getOperand(2).getImm();
    break;
  }
  if (!AddIndex)
    return std::nullopt;

  for (size_t I = *AddIndex; I != 0;) {
    --I;
    const InternalDecodedInst &Candidate = Decoded[I];
    if (!Candidate.DecodeSucceeded || isControlFlowBoundary(Candidate, LS))
      return std::nullopt;
    if (!definesOverlappingRegister(Candidate, LS, TargetRegister))
      continue;
    if (Candidate.Inst.getOpcode() != LS.SGetPcI64Opcode ||
        Candidate.Inst.getNumOperands() != 1 ||
        !Candidate.Inst.getOperand(0).isReg() ||
        Candidate.Inst.getOperand(0).getReg() != TargetRegister)
      return std::nullopt;

    std::optional<uint64_t> GetPcAddress = checkedAddUint64(
        TextAddr, Candidate.Offset, "PC-materialized call instruction");
    if (!GetPcAddress)
      return std::nullopt;
    std::optional<uint64_t> PcValue = checkedAddUint64(
        *GetPcAddress, Candidate.Size, "PC-materialized call PC value");
    if (!PcValue)
      return std::nullopt;
    // s_add_nc_u64 uses modulo-2^64 arithmetic. Casting the signed MC
    // immediate to uint64_t and adding it reproduces both positive and
    // negative literals, including INT64_MIN, without signed overflow.
    return PcMaterializedCallInfo{
        *PcValue + static_cast<uint64_t>(AddImmediate), Candidate.Offset,
        Call.Offset, MCRegister(Call.Inst.getOperand(0).getReg())};
  }
  return std::nullopt;
}

struct KnownCallSite {
  size_t InstIndex = 0;
  uint64_t Target = 0;
  uint64_t Continuation = 0;
  MCRegister ReturnRegister;
};

struct BoundedSetPcReturn {
  size_t InstIndex = 0;
  SmallVector<uint64_t, 2> Targets;
};

static bool hasPcRelativeOperand(const InternalDecodedInst &DI,
                                 const LLVMState &LS) {
  for (const MCOperandInfo &Operand :
       LS.MCII->get(DI.Inst.getOpcode()).operands())
    if (Operand.OperandType == MCOI::OPERAND_PCREL)
      return true;
  return false;
}

static std::optional<MCRegister>
getCallReturnRegister(const InternalDecodedInst &DI, const LLVMState &LS) {
  if (!DI.DecodeSucceeded || !LS.MIA->isCall(DI.Inst) ||
      DI.Inst.getNumOperands() == 0 || !DI.Inst.getOperand(0).isReg() ||
      !DI.Inst.getOperand(0).getReg())
    return std::nullopt;
  return MCRegister(DI.Inst.getOperand(0).getReg());
}

static std::optional<uint64_t>
getDirectTextTarget(const InternalDecodedInst &DI, const LLVMState &LS,
                    uint64_t TextAddr, uint64_t TextEnd) {
  if (!DI.DecodeSucceeded ||
      (!LS.MIA->isBranch(DI.Inst) && !LS.MIA->isCall(DI.Inst)) ||
      LS.MIA->isReturn(DI.Inst) || LS.MIA->isIndirectBranch(DI.Inst))
    return std::nullopt;

  if (hasPcRelativeOperand(DI, LS))
    return evaluateDirectControlFlowTarget(DI, LS);

  if (DI.Inst.getOpcode() != LS.SSwapPcI64Opcode ||
      DI.Inst.getNumOperands() == 0 ||
      !DI.Inst.getOperand(DI.Inst.getNumOperands() - 1).isImm())
    return std::nullopt;
  uint64_t AbsoluteTarget = static_cast<uint64_t>(
      DI.Inst.getOperand(DI.Inst.getNumOperands() - 1).getImm());
  if (AbsoluteTarget < TextAddr || AbsoluteTarget >= TextEnd)
    return std::nullopt;
  return AbsoluteTarget - TextAddr;
}

static std::optional<SmallVector<KnownCallSite, 4>> collectKnownCallSites(
    ArrayRef<InternalDecodedInst> Decoded, const LLVMState &LS,
    uint64_t TextAddr, uint64_t TextEnd,
    ArrayRef<std::optional<PcMaterializedCallInfo>> MaterializedCalls) {
  SmallVector<KnownCallSite, 4> Calls;
  for (size_t I = 0; I != Decoded.size(); ++I) {
    const InternalDecodedInst &DI = Decoded[I];
    std::optional<MCRegister> ReturnRegister = getCallReturnRegister(DI, LS);
    if (!ReturnRegister)
      continue;

    std::optional<uint64_t> Target;
    if (MaterializedCalls[I]) {
      uint64_t AbsoluteTarget = MaterializedCalls[I]->Target;
      if (AbsoluteTarget >= TextAddr && AbsoluteTarget < TextEnd)
        Target = AbsoluteTarget - TextAddr;
    } else {
      Target = getDirectTextTarget(DI, LS, TextAddr, TextEnd);
    }
    if (!Target)
      continue;

    std::optional<uint64_t> Continuation =
        checkedAddUint64(DI.Offset, DI.Size, "known call continuation address");
    if (!Continuation)
      return std::nullopt;
    Calls.push_back({I, *Target, *Continuation, *ReturnRegister});
  }
  return Calls;
}

static bool hasUnprovenFallthroughEntry(ArrayRef<InternalDecodedInst> Decoded,
                                        const LLVMState &LS, uint64_t TextAddr,
                                        uint64_t TextEnd,
                                        uint64_t FunctionBegin,
                                        uint64_t ReturnOffset,
                                        ArrayRef<uint64_t> DeclaredEntries,
                                        ArrayRef<KnownCallSite> Calls) {
  if (FunctionBegin == 0)
    return false;

  size_t BeginIndex = 0;
  while (BeginIndex != Decoded.size() &&
         Decoded[BeginIndex].Offset < FunctionBegin)
    ++BeginIndex;
  if (BeginIndex == Decoded.size() ||
      Decoded[BeginIndex].Offset != FunctionBegin) {
    log() << "hotswap: s_set_pc_i64 at 0x" << utohexstr(ReturnOffset)
          << " is not a bounded return: function entry at 0x"
          << utohexstr(FunctionBegin) << " is not an instruction boundary\n";
    return true;
  }

  uint64_t ChainBegin = FunctionBegin;
  size_t PredecessorIndex = BeginIndex;
  while (PredecessorIndex != 0) {
    const InternalDecodedInst &Predecessor = Decoded[PredecessorIndex - 1];
    std::optional<uint64_t> PredecessorEnd = checkedAddUint64(
        Predecessor.Offset, Predecessor.Size, "fallthrough predecessor end");
    if (!PredecessorEnd || *PredecessorEnd != ChainBegin ||
        !Predecessor.DecodeSucceeded) {
      log() << "hotswap: s_set_pc_i64 at 0x" << utohexstr(ReturnOffset)
            << " is not a bounded return: fallthrough into function entry "
               "at 0x"
            << utohexstr(FunctionBegin) << " is unprovable\n";
      return true;
    }
    if (LS.MIA->isBarrier(Predecessor.Inst))
      break;
    ChainBegin = Predecessor.Offset;
    --PredecessorIndex;
  }

  if (ChainBegin == FunctionBegin)
    return false;

  for (uint64_t Entry : DeclaredEntries)
    if (Entry >= ChainBegin && Entry < FunctionBegin) {
      log() << "hotswap: s_set_pc_i64 at 0x" << utohexstr(ReturnOffset)
            << " is not a bounded return: declared entry at 0x"
            << utohexstr(Entry) << " falls through to function entry 0x"
            << utohexstr(FunctionBegin) << "\n";
      return true;
    }

  for (const KnownCallSite &Call : Calls) {
    uint64_t Source = Decoded[Call.InstIndex].Offset;
    if (Source >= ChainBegin && Source < FunctionBegin)
      continue;
    if ((Call.Target >= ChainBegin && Call.Target < FunctionBegin) ||
        (Call.Continuation >= ChainBegin &&
         Call.Continuation < FunctionBegin)) {
      log() << "hotswap: s_set_pc_i64 at 0x" << utohexstr(ReturnOffset)
            << " is not a bounded return: call at 0x" << utohexstr(Source)
            << " enters the fallthrough chain at 0x"
            << utohexstr(Call.Target >= ChainBegin &&
                                 Call.Target < FunctionBegin
                             ? Call.Target
                             : Call.Continuation)
            << "\n";
      return true;
    }
  }

  for (const InternalDecodedInst &Source : Decoded) {
    std::optional<uint64_t> Target =
        getDirectTextTarget(Source, LS, TextAddr, TextEnd);
    if (!Target || *Target < ChainBegin || *Target >= FunctionBegin ||
        (Source.Offset >= ChainBegin && Source.Offset < FunctionBegin))
      continue;
    log() << "hotswap: s_set_pc_i64 at 0x" << utohexstr(ReturnOffset)
          << " is not a bounded return: control flow at 0x"
          << utohexstr(Source.Offset) << " enters the fallthrough chain at 0x"
          << utohexstr(*Target) << "\n";
    return true;
  }
  return false;
}

static std::optional<SmallVector<BoundedSetPcReturn, 2>>
collectBoundedSetPcReturns(ArrayRef<InternalDecodedInst> Decoded,
                           const LLVMState &LS, uint64_t TextAddr,
                           uint64_t TextEnd, ArrayRef<uint64_t> DeclaredEntries,
                           ArrayRef<ElfView::FunctionTextRange> FunctionRanges,
                           ArrayRef<KnownCallSite> Calls,
                           ArrayRef<uint64_t> ExternalEntries) {
  SmallVector<BoundedSetPcReturn, 2> Returns;
  for (size_t ReturnIndex = 0; ReturnIndex != Decoded.size(); ++ReturnIndex) {
    const InternalDecodedInst &Return = Decoded[ReturnIndex];
    // AMDGPUMCInstLower lowers S_SETPC_B64_return to S_SETPC_B64, so the
    // decoded instruction no longer carries MIA::isReturn identity. Recover
    // only the bounded local-function form from its call/link dataflow.
    if (!Return.DecodeSucceeded ||
        Return.Inst.getOpcode() != LS.SSetPcI64Opcode ||
        Return.Inst.getNumOperands() != 1 ||
        !Return.Inst.getOperand(0).isReg() ||
        !Return.Inst.getOperand(0).getReg())
      continue;
    MCRegister ReturnRegister(Return.Inst.getOperand(0).getReg());

    bool IsBounded = false;
    for (const ElfView::FunctionTextRange &Range : FunctionRanges) {
      if (Range.Begin < TextAddr || Range.Begin >= TextEnd ||
          Range.End <= Range.Begin || Range.End > TextEnd ||
          (Range.Symbol && Range.Symbol->getBinding() != ELF::STB_LOCAL))
        continue;
      uint64_t FunctionBegin = Range.Begin - TextAddr;
      uint64_t FunctionEnd = Range.End - TextAddr;
      if (Return.Offset < FunctionBegin || Return.Offset >= FunctionEnd)
        continue;

      bool Safe = true;
      for (uint64_t Entry : ExternalEntries)
        if (Entry >= FunctionBegin && Entry < FunctionEnd) {
          log() << "hotswap: s_set_pc_i64 at 0x" << utohexstr(Return.Offset)
                << " is not a bounded return: externally reachable entry at "
                   "0x"
                << utohexstr(Entry) << " overlaps the local function\n";
          Safe = false;
          break;
        }
      if (!Safe)
        continue;

      for (const ElfView::FunctionTextRange &Alias : FunctionRanges) {
        if (Return.Offset + TextAddr < Alias.Begin ||
            Return.Offset + TextAddr >= Alias.End || Alias.Begin == Range.Begin)
          continue;
        log() << "hotswap: s_set_pc_i64 at 0x" << utohexstr(Return.Offset)
              << " is not a bounded return: overlapping function entry at "
                 "0x"
              << utohexstr(Alias.Begin - TextAddr)
              << " makes entry provenance ambiguous\n";
        Safe = false;
        break;
      }
      if (!Safe)
        continue;

      for (uint64_t Entry : DeclaredEntries)
        if (Entry > FunctionBegin && Entry < FunctionEnd) {
          log() << "hotswap: s_set_pc_i64 at 0x" << utohexstr(Return.Offset)
                << " is not a bounded return: declared entry at 0x"
                << utohexstr(Entry) << " bypasses the function entry\n";
          Safe = false;
          break;
        }
      if (!Safe)
        continue;

      if (hasUnprovenFallthroughEntry(Decoded, LS, TextAddr, TextEnd,
                                      FunctionBegin, Return.Offset,
                                      DeclaredEntries, Calls))
        continue;

      // The link pair must retain the value written by the incoming call
      // throughout the function. This includes blocks laid out after the
      // return that may branch back into its epilogue.
      for (const InternalDecodedInst &DI : Decoded) {
        if (DI.Offset < FunctionBegin || DI.Offset >= FunctionEnd)
          continue;
        // MC call instructions carry no transitive callee-clobber information.
        // Without interprocedural proof, a nested callee may overwrite the
        // outer link pair even when the call defines a different return pair.
        if (DI.DecodeSucceeded && LS.MIA->isCall(DI.Inst)) {
          log() << "hotswap: s_set_pc_i64 at 0x" << utohexstr(Return.Offset)
                << " is not a bounded return: nested call at 0x"
                << utohexstr(DI.Offset) << " may clobber the link register\n";
          Safe = false;
          break;
        }
        if (!DI.DecodeSucceeded ||
            definesOverlappingRegister(DI, LS, ReturnRegister)) {
          log() << "hotswap: s_set_pc_i64 at 0x" << utohexstr(Return.Offset)
                << " is not a bounded return: link register is "
                   "unprovable at 0x"
                << utohexstr(DI.Offset) << "\n";
          Safe = false;
          break;
        }
      }
      if (!Safe)
        continue;

      SmallVector<uint64_t, 2> Targets;
      for (const KnownCallSite &Call : Calls) {
        if (Call.Target < FunctionBegin || Call.Target >= FunctionEnd)
          continue;
        if (Call.Target != FunctionBegin) {
          log() << "hotswap: s_set_pc_i64 at 0x" << utohexstr(Return.Offset)
                << " is not a bounded return: call at 0x"
                << utohexstr(Decoded[Call.InstIndex].Offset)
                << " enters the function interior at 0x"
                << utohexstr(Call.Target) << "\n";
          Safe = false;
          break;
        }
        if (Call.ReturnRegister != ReturnRegister) {
          log() << "hotswap: s_set_pc_i64 at 0x" << utohexstr(Return.Offset)
                << " is not a bounded return: call at 0x"
                << utohexstr(Decoded[Call.InstIndex].Offset)
                << " uses a different link register\n";
          Safe = false;
          break;
        }
        Targets.push_back(Call.Continuation);
      }
      if (!Safe || Targets.empty())
        continue;

      // A branch from outside the function would bypass the call link
      // definition. Direct calls to the function entry are allowed only when
      // they were collected above with this exact return register.
      for (size_t SourceIndex = 0; SourceIndex != Decoded.size();
           ++SourceIndex) {
        const InternalDecodedInst &Source = Decoded[SourceIndex];
        std::optional<uint64_t> Target =
            getDirectTextTarget(Source, LS, TextAddr, TextEnd);
        if (!Target || *Target < FunctionBegin || *Target >= FunctionEnd ||
            (Source.Offset >= FunctionBegin && Source.Offset < FunctionEnd))
          continue;

        bool IsKnownEntryCall = false;
        if (LS.MIA->isCall(Source.Inst) && *Target == FunctionBegin)
          for (const KnownCallSite &Call : Calls)
            IsKnownEntryCall |= Call.InstIndex == SourceIndex &&
                                Call.ReturnRegister == ReturnRegister;
        if (!IsKnownEntryCall) {
          log() << "hotswap: s_set_pc_i64 at 0x" << utohexstr(Return.Offset)
                << " is not a bounded return: control flow at 0x"
                << utohexstr(Source.Offset) << " enters at 0x"
                << utohexstr(*Target) << "\n";
          Safe = false;
          break;
        }
      }
      if (!Safe)
        continue;

      llvm::sort(Targets);
      Targets.erase(std::unique(Targets.begin(), Targets.end()), Targets.end());
      Returns.push_back({ReturnIndex, std::move(Targets)});
      IsBounded = true;
      break;
    }
    if (!IsBounded)
      log() << "hotswap: s_set_pc_i64 at 0x" << utohexstr(Return.Offset)
            << " remains an unbounded indirect transfer\n";
  }
  return Returns;
}

static const BoundedSetPcReturn *
findBoundedSetPcReturn(ArrayRef<BoundedSetPcReturn> Returns, size_t InstIndex) {
  for (const BoundedSetPcReturn &Return : Returns)
    if (Return.InstIndex == InstIndex)
      return &Return;
  return nullptr;
}

static bool
hasKnownControlFlowEntry(ArrayRef<InternalDecodedInst> Decoded,
                         const LLVMState &LS, uint64_t TextAddr,
                         uint64_t TextEnd, ArrayRef<uint64_t> DeclaredEntries,
                         ArrayRef<BoundedSetPcReturn> BoundedReturns,
                         uint64_t SequenceStart, uint64_t SequenceEnd) {
  for (uint64_t Entry : DeclaredEntries)
    if (Entry > SequenceStart && Entry <= SequenceEnd)
      return true;

  for (size_t I = 0; I != Decoded.size(); ++I) {
    const InternalDecodedInst &DI = Decoded[I];
    if (DI.DecodeSucceeded && DI.Inst.getOpcode() == LS.SSetPcI64Opcode) {
      const BoundedSetPcReturn *Return =
          findBoundedSetPcReturn(BoundedReturns, I);
      if (!Return)
        return true;
      for (uint64_t Target : Return->Targets)
        if (Target > SequenceStart && Target <= SequenceEnd)
          return true;
      continue;
    }

    // Without bounding an indirect target, it may enter at any instruction in
    // the materialization. Keep the call unresolved rather than relying on
    // the indirect transfer's containing function alone.
    if (DI.DecodeSucceeded && !LS.MIA->isReturn(DI.Inst) &&
        (LS.MIA->isIndirectBranch(DI.Inst) ||
         DI.Inst.getOpcode() == LS.SAddPcI64Opcode))
      return true;
    if ((!LS.MIA->isBranch(DI.Inst) && !LS.MIA->isCall(DI.Inst)) ||
        LS.MIA->isReturn(DI.Inst))
      continue;

    std::optional<uint64_t> RelativeTarget =
        getDirectTextTarget(DI, LS, TextAddr, TextEnd);
    if (RelativeTarget && *RelativeTarget > SequenceStart &&
        *RelativeTarget <= SequenceEnd)
      return true;
  }
  return false;
}

/// Collect statically known direct branch and call destinations so an interior
/// entry point is never swallowed by coalescing.
std::optional<DirectControlFlowInfo> collectDirectBranchTargets(
    ArrayRef<InternalDecodedInst> Decoded, const LLVMState &LS,
    uint64_t TextAddr, uint64_t TextSize, ArrayRef<uint64_t> DeclaredEntries,
    ArrayRef<ElfView::FunctionTextRange> FunctionRanges,
    ArrayRef<uint64_t> ExternalEntries) {
  if (!LS.MIA) {
    log() << "hotswap: MC branch analysis is unavailable; adjacent far "
             "trampolines will not be coalesced\n";
    return std::nullopt;
  }

  std::optional<uint64_t> TextEnd =
      checkedAddUint64(TextAddr, TextSize, "direct target text end");
  if (!TextEnd)
    return std::nullopt;

  SmallVector<std::optional<PcMaterializedCallInfo>, 16> MaterializedCalls(
      Decoded.size());
  for (size_t I = 0; I != Decoded.size(); ++I)
    MaterializedCalls[I] = matchPcMaterializedCall(Decoded, I, LS, TextAddr);

  std::optional<SmallVector<KnownCallSite, 4>> Calls =
      collectKnownCallSites(Decoded, LS, TextAddr, *TextEnd, MaterializedCalls);
  if (!Calls)
    return std::nullopt;
  std::optional<SmallVector<BoundedSetPcReturn, 2>> BoundedReturns =
      collectBoundedSetPcReturns(Decoded, LS, TextAddr, *TextEnd,
                                 DeclaredEntries, FunctionRanges, *Calls,
                                 ExternalEntries);
  if (!BoundedReturns)
    return std::nullopt;

  DirectControlFlowInfo Info;
  for (size_t InstIndex = 0; InstIndex != Decoded.size(); ++InstIndex) {
    const InternalDecodedInst &DI = Decoded[InstIndex];
    if ((!LS.MIA->isBranch(DI.Inst) && !LS.MIA->isCall(DI.Inst)) ||
        LS.MIA->isReturn(DI.Inst))
      continue;
    // Existing indirect branches are handled by
    // collectIndirectControlFlowFunctions(), which protects their containing
    // function from source relocation. Calls without a statically resolvable
    // target are handled below.
    if (LS.MIA->isIndirectBranch(DI.Inst))
      continue;

    bool HasPcRelativeOperand = false;
    for (const MCOperandInfo &Op : LS.MCII->get(DI.Inst.getOpcode()).operands())
      HasPcRelativeOperand |= Op.OperandType == MCOI::OPERAND_PCREL;
    if (!HasPcRelativeOperand) {
      // Preserve the established handling for non-call indirect transfers
      // such as s_set_pc_i64. collectIndirectControlFlowFunctions() prevents
      // source relocation in their containing function.
      if (!LS.MIA->isCall(DI.Inst))
        continue;
      std::optional<uint64_t> Target;
      if (DI.Inst.getOpcode() == LS.SSwapPcI64Opcode &&
          DI.Inst.getNumOperands() != 0 &&
          DI.Inst.getOperand(DI.Inst.getNumOperands() - 1).isImm()) {
        Target = static_cast<uint64_t>(
            DI.Inst.getOperand(DI.Inst.getNumOperands() - 1).getImm());
      } else if (MaterializedCalls[InstIndex] &&
                 !hasKnownControlFlowEntry(
                     Decoded, LS, TextAddr, *TextEnd, DeclaredEntries,
                     *BoundedReturns,
                     MaterializedCalls[InstIndex]->SequenceStart,
                     MaterializedCalls[InstIndex]->SequenceEnd)) {
        Target = MaterializedCalls[InstIndex]->Target;
      }
      if (!Target) {
        log() << "hotswap: unresolved call target at 0x" << utohexstr(DI.Offset)
              << " (" << DI.Mnemonic << ")\n";
        Info.HasUnresolvedTargets = true;
        continue;
      }

      if (*Target >= TextAddr && *Target < *TextEnd) {
        uint64_t RelativeTarget = *Target - TextAddr;
        Info.Targets.insert(RelativeTarget);
        if (DI.Inst.getOperand(DI.Inst.getNumOperands() - 1).isReg())
          log() << "hotswap: resolved PC-materialized call at 0x"
                << utohexstr(DI.Offset) << " to .text+0x"
                << utohexstr(RelativeTarget) << "\n";
      }
      continue;
    }

    std::optional<uint64_t> Target = evaluateDirectControlFlowTarget(DI, LS);
    if (!Target) {
      log() << "hotswap: MC analysis could not evaluate direct control-flow "
               "instruction at 0x"
            << utohexstr(DI.Offset)
            << "; adjacent far trampolines will not be coalesced\n";
      return std::nullopt;
    }
    Info.Targets.insert(*Target);
  }
  return Info;
}

/// Coalesce runs of adjacent far patch sites when the same SGPR scratch block
/// is safe at every site. Removing each interior return reservation preserves
/// replacement order and reduces the number of required forward gateways.
/// This deliberately never steals an unpatched neighboring instruction.
static void
mergeAdjacentLongTrampolines(std::vector<Trampoline> &Trampolines,
                             const DenseSet<uint64_t> &DirectBranchTargets) {
  std::vector<Trampoline> Merged;
  Merged.reserve(Trampolines.size());
  uint64_t MergeCount = 0;

  for (Trampoline &T : Trampolines) {
    bool Adjacent = false;
    if (!Merged.empty()) {
      Trampoline &Prev = Merged.back();
      std::optional<uint64_t> PrevEnd = checkedAddUint64(
          Prev.OriginalOffset, Prev.OriginalSize, "adjacent trampoline end");
      Adjacent = PrevEnd && *PrevEnd == T.OriginalOffset && Prev.Long &&
                 T.Long && Prev.UsesSetPCBack && T.UsesSetPCBack &&
                 Prev.LongBranchSgprBase == T.LongBranchSgprBase &&
                 Prev.HasFunctionRange && T.HasFunctionRange &&
                 Prev.FunctionStart == T.FunctionStart &&
                 Prev.FunctionEnd == T.FunctionEnd &&
                 !DirectBranchTargets.contains(T.OriginalOffset) &&
                 Prev.Bytes.size() >= SetPcReturnReserveBytes &&
                 T.Bytes.size() >= SetPcReturnReserveBytes;
    }

    if (!Adjacent) {
      Merged.emplace_back(std::move(T));
      continue;
    }

    Trampoline &Prev = Merged.back();
    if (T.OriginalSize >
        std::numeric_limits<uint32_t>::max() - Prev.OriginalSize) {
      Merged.emplace_back(std::move(T));
      continue;
    }
    Prev.Bytes.resize(Prev.Bytes.size() - SetPcReturnReserveBytes);
    Prev.Bytes.append(T.Bytes.begin(), T.Bytes.end());
    Prev.OriginalSize += T.OriginalSize;
    ++MergeCount;
  }

  Trampolines = std::move(Merged);
  if (MergeCount != 0)
    log() << "hotswap: coalesced " << MergeCount
          << " adjacent far trampoline edge(s)\n";
}

static void appendPoolBranchIslands(std::vector<Trampoline> &Trampolines) {
  for (Trampoline &T : Trampolines) {
    if (!T.Long)
      continue;
    T.Bytes.append(PoolBranchIslandBytes, uint8_t{0});
    T.HasPoolBranchIsland = true;
  }
}

static bool isEndProgram(const InternalDecodedInst &DI, const LLVMState &LS) {
  unsigned Opcode = DI.Inst.getOpcode();
  return Opcode == LS.SEndPgmOpcode || Opcode == LS.SEndPgmSavedOpcode;
}

static bool isPcSensitive(const InternalDecodedInst &DI, const LLVMState &LS) {
  unsigned Opcode = DI.Inst.getOpcode();
  return Opcode == LS.SAddPcI64Opcode || Opcode == LS.SGetPcI64Opcode ||
         Opcode == LS.SSetPcI64Opcode || Opcode == LS.SSwapPcI64Opcode ||
         Opcode == LS.SPrefetchInstPcRelOpcode ||
         Opcode == LS.SPrefetchDataPcRelOpcode;
}

static bool isSafeStraightLineRelocation(const InternalDecodedInst &DI,
                                         const LLVMState &LS,
                                         const DenseSet<uint64_t> &Protected) {
  if (!LS.MIA || LS.MIA->mayAffectControlFlow(DI.Inst, *LS.MRI))
    return false;
  unsigned Opcode = DI.Inst.getOpcode();
  return DI.DecodeSucceeded && !Protected.contains(DI.Offset) &&
         Opcode != LS.SClauseOpcode && Opcode != LS.SDelayAluOpcode &&
         !isPcSensitive(DI, LS);
}

/// Decode the bytes currently present at an original instruction site. Earlier
/// rewrite passes may have changed Ctx.Text after Ctx.Decoded was populated, so
/// relocation decisions must not classify the stale MCInst and then copy a
/// different instruction. A size change is conservatively non-relocatable.
static std::optional<InternalDecodedInst>
decodeCurrentInstruction(const PatchContext &Ctx,
                         const InternalDecodedInst &Original) {
  if (Original.Offset > Ctx.TextSize ||
      Original.Size > Ctx.TextSize - Original.Offset)
    return std::nullopt;

  std::vector<InternalDecodedInst> Current;
  if (!decodeTextSection(Ctx.Text + Original.Offset, Original.Size, Ctx.LS,
                         Current) ||
      Current.size() != 1 || Current[0].Size != Original.Size)
    return std::nullopt;
  Current[0].Offset = Original.Offset;
  return std::move(Current[0]);
}

/// Instructions covered by a hard clause or a delay directive must remain in
/// place relative to that directive. B0-to-A0 rewrites have already replaced
/// clauses with s_nop, so only preserve clause members when requested. Always
/// mark the maximum six-instruction forward span addressable by s_delay_alu.
static DenseSet<uint64_t>
collectRelocationProtectedOffsets(ArrayRef<InternalDecodedInst> Decoded,
                                  const LLVMState &LS,
                                  bool ProtectClauseMembers) {
  DenseSet<uint64_t> Protected;
  unsigned ClauseRemaining = 0;
  unsigned DelayRemaining = 0;

  for (const InternalDecodedInst &DI : Decoded) {
    if (ClauseRemaining != 0) {
      Protected.insert(DI.Offset);
      --ClauseRemaining;
    }
    if (DelayRemaining != 0) {
      Protected.insert(DI.Offset);
      --DelayRemaining;
    }

    if (ProtectClauseMembers && DI.Inst.getOpcode() == LS.SClauseOpcode &&
        DI.Inst.getNumOperands() == 1 && DI.Inst.getOperand(0).isImm())
      ClauseRemaining =
          (static_cast<unsigned>(DI.Inst.getOperand(0).getImm()) & 63u) + 1;
    else if (DI.Inst.getOpcode() == LS.SDelayAluOpcode)
      DelayRemaining = 6;
  }
  return Protected;
}

/// Relocating an instruction changes its address. In a function containing a
/// register-based PC transfer, MC cannot prove that the instruction is not an
/// indirect destination, so leave the complete function in place.
static DenseSet<uint64_t>
collectIndirectControlFlowFunctions(ArrayRef<InternalDecodedInst> Decoded,
                                    const LLVMState &LS, const ElfView &Elf) {
  DenseSet<uint64_t> Functions;
  if (!LS.MIA)
    return Functions;

  for (const InternalDecodedInst &DI : Decoded) {
    if (LS.MIA->isBarrier(DI.Inst) || isEndProgram(DI, LS))
      continue;
    if (!LS.MIA->isIndirectBranch(DI.Inst) &&
        !(LS.MIA->mayAffectControlFlow(DI.Inst, *LS.MRI) &&
          isPcSensitive(DI, LS)))
      continue;
    std::optional<ElfView::FunctionTextRange> Range =
        Elf.findFunctionTextRangeAtOffset(DI.Offset);
    if (Range && Functions.insert(Range->Begin).second)
      log() << "hotswap: source relocation disabled for function at 0x"
            << utohexstr(Range->Begin) << " by " << DI.Mnemonic << " at 0x"
            << utohexstr(DI.Offset) << "\n";
  }
  return Functions;
}

/// Grow undersized far-site windows only through proven straight-line code.
/// Patched neighbors are merged; ordinary instructions are copied verbatim
/// into the trampoline body and retain their original order. This is bounded
/// to the 20 bytes required by the gfx12 SCC-neutral forward sequence.
static void
expandStraightLineTrampolines(PatchContext &Ctx,
                              const DenseSet<uint64_t> &DirectBranchTargets) {
  DenseMap<uint64_t, size_t> DecodedAt;
  for (size_t I = 0; I != Ctx.Decoded.size(); ++I)
    DecodedAt[Ctx.Decoded[I].Offset] = I;
  DenseSet<uint64_t> Protected = collectRelocationProtectedOffsets(
      Ctx.Decoded, Ctx.LS, !Ctx.Config.RunB0A0Patches);
  DenseSet<uint64_t> IndirectControlFlowFunctions =
      collectIndirectControlFlowFunctions(Ctx.Decoded, Ctx.LS, Ctx.Elf);

  for (size_t I = 0; I != Ctx.OutTrampolines.size(); ++I) {
    if (Ctx.OutTrampolines[I].HasFunctionRange &&
        IndirectControlFlowFunctions.contains(
            Ctx.OutTrampolines[I].FunctionStart))
      continue;
    while (Ctx.OutTrampolines[I].Long &&
           Ctx.OutTrampolines[I].OriginalSize < SetPcForwardSequenceBytes) {
      Trampoline &T = Ctx.OutTrampolines[I];
      std::optional<uint64_t> End = checkedAddUint64(
          T.OriginalOffset, T.OriginalSize, "straight-line expansion end");
      if (!End || DirectBranchTargets.contains(*End))
        break;

      if (I + 1 < Ctx.OutTrampolines.size() &&
          Ctx.OutTrampolines[I + 1].OriginalOffset == *End) {
        Trampoline &Next = Ctx.OutTrampolines[I + 1];
        if (!Next.Long || !Next.UsesSetPCBack ||
            Next.LongBranchSgprBase != T.LongBranchSgprBase ||
            !T.HasFunctionRange || !Next.HasFunctionRange ||
            T.FunctionStart != Next.FunctionStart ||
            T.FunctionEnd != Next.FunctionEnd ||
            Next.Bytes.size() < SetPcReturnReserveBytes)
          break;
        T.Bytes.resize(T.Bytes.size() - SetPcReturnReserveBytes);
        T.Bytes.append(Next.Bytes.begin(), Next.Bytes.end());
        T.OriginalSize += Next.OriginalSize;
        Ctx.OutTrampolines.erase(Ctx.OutTrampolines.begin() + I + 1);
        continue;
      }

      DenseMap<uint64_t, size_t>::const_iterator It = DecodedAt.find(*End);
      if (It == DecodedAt.end())
        break;
      const InternalDecodedInst &Original = Ctx.Decoded[It->second];
      std::optional<InternalDecodedInst> Current =
          decodeCurrentInstruction(Ctx, Original);
      if (!Current)
        break;
      const InternalDecodedInst &DI = *Current;
      std::optional<ElfView::FunctionTextRange> Range =
          Ctx.Elf.findFunctionTextRangeAtOffset(DI.Offset);
      if (!Range || !T.HasFunctionRange || Range->Begin != T.FunctionStart ||
          Range->End != T.FunctionEnd ||
          !isSafeStraightLineRelocation(DI, Ctx.LS, Protected) ||
          T.Bytes.size() < SetPcReturnReserveBytes)
        break;

      T.Bytes.insert(T.Bytes.end() - SetPcReturnReserveBytes,
                     Ctx.Text + DI.Offset, Ctx.Text + DI.Offset + DI.Size);
      T.OriginalSize += DI.Size;
    }

    while (Ctx.OutTrampolines[I].Long &&
           Ctx.OutTrampolines[I].OriginalSize < SetPcForwardSequenceBytes) {
      Trampoline &T = Ctx.OutTrampolines[I];
      if (DirectBranchTargets.contains(T.OriginalOffset))
        break;
      DenseMap<uint64_t, size_t>::const_iterator It =
          DecodedAt.find(T.OriginalOffset);
      if (It == DecodedAt.end() || It->second == 0)
        break;
      const InternalDecodedInst &Original = Ctx.Decoded[It->second - 1];
      std::optional<InternalDecodedInst> Current =
          decodeCurrentInstruction(Ctx, Original);
      if (!Current)
        break;
      const InternalDecodedInst &DI = *Current;
      if (DI.Offset + DI.Size != T.OriginalOffset ||
          !isSafeStraightLineRelocation(DI, Ctx.LS, Protected))
        break;
      if (I != 0) {
        const Trampoline &Previous = Ctx.OutTrampolines[I - 1];
        if (Previous.OriginalOffset + Previous.OriginalSize > DI.Offset)
          break;
      }
      std::optional<ElfView::FunctionTextRange> Range =
          Ctx.Elf.findFunctionTextRangeAtOffset(DI.Offset);
      if (!Range || !T.HasFunctionRange || Range->Begin != T.FunctionStart ||
          Range->End != T.FunctionEnd)
        break;
      T.Bytes.insert(T.Bytes.begin(), Ctx.Text + DI.Offset,
                     Ctx.Text + DI.Offset + DI.Size);
      T.OriginalOffset = DI.Offset;
      T.OriginalSize += DI.Size;
    }
  }
}

static bool hasNoFallthrough(const InternalDecodedInst &DI,
                             const LLVMState &LS) {
  return isEndProgram(DI, LS) ||
         (LS.MIA &&
          (LS.MIA->isUnconditionalBranch(DI.Inst) ||
           LS.MIA->isReturn(DI.Inst) || LS.MIA->isIndirectBranch(DI.Inst) ||
           LS.MIA->isBarrier(DI.Inst)));
}

static void appendGatewaySled(std::vector<NopSled> &Sleds, uint64_t Start,
                              uint64_t End, uint64_t TextSize, bool Safe,
                              bool HasTarget) {
  if (Safe && !HasTarget && End - Start >= MinInstSize)
    Sleds.push_back({Start, End, Start, 0, TextSize});
}

/// Find zero-filled alignment holes, including holes covered by an oversized
/// function symbol, and s_nop padding outside every function. Such padding is
/// a safe branch gateway only when it follows a no-fallthrough instruction and
/// contains no direct branch/call target. In-function s_nop runs are added from
/// Ctx.NopSleds separately.
static std::vector<NopSled>
buildExternalGatewaySleds(ArrayRef<InternalDecodedInst> Decoded,
                          const LLVMState &LS, const ElfView &Elf,
                          ArrayRef<uint8_t> Text,
                          const DenseSet<uint64_t> &DirectBranchTargets) {
  std::vector<NopSled> Sleds;
  const InternalDecodedInst *Previous = nullptr;
  bool Active = false;
  bool Safe = false;
  bool HasTarget = false;
  uint64_t Start = 0;
  uint64_t End = 0;

  for (const InternalDecodedInst &DI : Decoded) {
    bool ZeroPadding =
        DI.Offset <= Text.size() && DI.Size <= Text.size() - DI.Offset;
    if (ZeroPadding)
      for (uint8_t Byte : Text.slice(DI.Offset, DI.Size))
        ZeroPadding &= Byte == 0;
    bool IsExternalNop = DI.Inst.getOpcode() == LS.SNopOpcode &&
                         !Elf.findFunctionTextRangeAtOffset(DI.Offset);
    bool GatewayPadding = ZeroPadding || IsExternalNop;
    if (!GatewayPadding || (Active && DI.Offset != End)) {
      if (Active)
        appendGatewaySled(Sleds, Start, End, Text.size(), Safe, HasTarget);
      Active = false;
    }
    if (!GatewayPadding) {
      Previous = &DI;
      continue;
    }
    if (!Active) {
      Active = true;
      Start = DI.Offset;
      Safe = Previous && hasNoFallthrough(*Previous, LS);
      HasTarget = false;
    }
    HasTarget |= DirectBranchTargets.contains(DI.Offset);
    End = DI.Offset + DI.Size;
  }
  if (Active)
    appendGatewaySled(Sleds, Start, End, Text.size(), Safe, HasTarget);
  return Sleds;
}

Expected<uint64_t>
countReachableSetPcGatewaySlots(ArrayRef<NopSled> Gateways, const LLVMState &LS,
                                uint64_t FromOffset, uint64_t TargetOffset,
                                unsigned SgprBase, uint64_t MaxSlots) {
  uint64_t Slots = 0;
  for (const NopSled &Sled : Gateways) {
    if (FromOffset < Sled.FunctionStart || FromOffset >= Sled.FunctionEnd)
      continue;
    uint64_t UsableEnd = std::min(Sled.End, Sled.FunctionEnd);
    uint64_t Candidate = Sled.WritePos;
    while (Candidate <= UsableEnd && Slots < MaxSlots) {
      uint64_t Distance = Candidate > FromOffset ? Candidate - FromOffset
                                                 : FromOffset - Candidate;
      if (Distance >= MaxSledDistance ||
          LS.encodeSBranch(FromOffset, Candidate).empty())
        break;
      std::optional<SmallVector<uint8_t>> Bytes =
          encodeSetPCLongBranch(LS, Candidate, TargetOffset, SgprBase);
      if (!Bytes)
        return createStringError(
            Twine("failed to encode set-PC gateway while counting candidate "
                  "offset 0x") +
            utohexstr(Candidate));
      if (Bytes->size() > UsableEnd - Candidate)
        break;
      ++Slots;
      Candidate += Bytes->size();
    }
    if (Slots == MaxSlots)
      break;
  }
  return Slots;
}

static std::optional<SmallVector<uint64_t, 4>>
allocateForwardBranchIslands(std::vector<NopSled> &Gateways,
                             uint64_t FromOffset, uint64_t TargetOffset) {
  struct Allocation {
    size_t SledIndex = 0;
    uint64_t PreviousWritePos = 0;
  };
  SmallVector<Allocation, 4> Allocations;
  SmallVector<uint64_t, 4> Islands;
  DenseSet<size_t> UsedSleds;
  uint64_t Current = FromOffset;

  while (!isSBranchReachable(Current, TargetOffset)) {
    size_t BestIndex = Gateways.size();
    uint64_t BestOffset = 0;
    for (size_t I = 0; I != Gateways.size(); ++I) {
      NopSled &Sled = Gateways[I];
      if (UsedSleds.contains(I) || FromOffset < Sled.FunctionStart ||
          FromOffset >= Sled.FunctionEnd)
        continue;
      uint64_t UsableEnd = std::min(Sled.End, Sled.FunctionEnd);
      if (Sled.WritePos >= TargetOffset || Sled.WritePos <= Current ||
          Sled.WritePos > UsableEnd ||
          MinInstSize > UsableEnd - Sled.WritePos ||
          !isSBranchReachable(Current, Sled.WritePos))
        continue;
      if (BestIndex == Gateways.size() || Sled.WritePos > BestOffset) {
        BestIndex = I;
        BestOffset = Sled.WritePos;
      }
    }

    if (BestIndex == Gateways.size()) {
      for (size_t I = Allocations.size(); I != 0; --I) {
        const Allocation &A = Allocations[I - 1];
        Gateways[A.SledIndex].WritePos = A.PreviousWritePos;
      }
      return std::nullopt;
    }

    NopSled &Best = Gateways[BestIndex];
    Allocations.push_back({BestIndex, Best.WritePos});
    Islands.push_back(Best.WritePos);
    Current = Best.WritePos;
    Best.WritePos += MinInstSize;
    UsedSleds.insert(BestIndex);
  }
  return Islands;
}

static bool
assignLongBranchGateways(PatchContext &Ctx,
                         const DenseSet<uint64_t> &DirectBranchTargets,
                         bool AllowTextGateways) {
  std::vector<NopSled> Gateways;
  if (AllowTextGateways) {
    Gateways = buildExternalGatewaySleds(
        Ctx.Decoded, Ctx.LS, Ctx.Elf, ArrayRef<uint8_t>(Ctx.Text, Ctx.TextSize),
        DirectBranchTargets);
    for (const NopSled &Sled : Ctx.NopSleds)
      Gateways.push_back(Sled);
  }

  DenseMap<uint64_t, size_t> PoolIslandOwners;
  uint64_t IslandLayoutOffset = Ctx.PoolBaseOffset;
  for (size_t I = 0; I != Ctx.OutTrampolines.size(); ++I) {
    Trampoline &T = Ctx.OutTrampolines[I];
    std::optional<uint64_t> Next = checkedAddUint64(
        IslandLayoutOffset, T.Bytes.size(), "pool branch-island layout");
    if (!Next)
      return false;
    if (T.HasPoolBranchIsland) {
      T.PoolBranchIslandOffset = *Next - PoolBranchIslandBytes;
      PoolIslandOwners[T.PoolBranchIslandOffset] = I;
      Gateways.push_back({T.PoolBranchIslandOffset,
                          T.PoolBranchIslandOffset + PoolBranchIslandBytes,
                          T.PoolBranchIslandOffset, 0,
                          std::numeric_limits<uint64_t>::max()});
    }
    IslandLayoutOffset = *Next;
  }

  struct PendingGateway {
    size_t TrampolineIndex = 0;
    uint64_t TargetOffset = 0;
    uint64_t InitialCandidateSlots = 0;
  };
  std::vector<PendingGateway> Pending;
  uint64_t TrampOffset = Ctx.PoolBaseOffset;
  for (size_t I = 0; I != Ctx.OutTrampolines.size(); ++I) {
    Trampoline &T = Ctx.OutTrampolines[I];
    uint64_t TP = TrampOffset;
    std::optional<uint64_t> Next = checkedAddUint64(
        TrampOffset, T.Bytes.size(), "gateway trampoline layout");
    if (!Next)
      return false;
    TrampOffset = *Next;
    if (!T.Long)
      continue;

    if (isSBranchReachable(T.OriginalOffset, TP)) {
      T.UsesShortBranchForward = true;
      continue;
    }
    std::optional<SmallVector<uint8_t>> Direct = encodeSetPCLongBranch(
        Ctx.LS, T.OriginalOffset, TP, T.LongBranchSgprBase);
    if (Direct && Direct->size() <= T.OriginalSize) {
      T.UsesDirectSetPCForward = true;
      T.DirectSetPCForwardBytes = std::move(*Direct);
      continue;
    }
    Pending.push_back({I, TP, 0});
  }
  for (PendingGateway &P : Pending) {
    const Trampoline &T = Ctx.OutTrampolines[P.TrampolineIndex];
    Expected<uint64_t> CandidateSlots = countReachableSetPcGatewaySlots(
        Gateways, Ctx.LS, T.OriginalOffset, P.TargetOffset,
        T.LongBranchSgprBase, Pending.size());
    if (!CandidateSlots) {
      log() << "hotswap: error: failed to count gateways for far site 0x"
            << utohexstr(T.OriginalOffset) << ": "
            << toString(CandidateSlots.takeError()) << "\n";
      return false;
    }
    P.InitialCandidateSlots = *CandidateSlots;
  }

  std::vector<PendingGateway> StillPending;
  StillPending.reserve(Pending.size());
  uint64_t BranchIslandChains = 0;
  for (const PendingGateway &P : Pending) {
    Trampoline &T = Ctx.OutTrampolines[P.TrampolineIndex];
    std::optional<SmallVector<uint64_t, 4>> Islands =
        allocateForwardBranchIslands(Gateways, T.OriginalOffset,
                                     P.TargetOffset);
    if (!Islands || Islands->empty()) {
      StillPending.push_back(P);
      continue;
    }
    T.ForwardBranchIslands = std::move(*Islands);
    T.ForwardBranchTargetOffset = P.TargetOffset;
    ++BranchIslandChains;
  }
  Pending = std::move(StillPending);

  std::stable_sort(Pending.begin(), Pending.end(),
                   [](const PendingGateway &LHS, const PendingGateway &RHS) {
                     return LHS.InitialCandidateSlots <
                            RHS.InitialCandidateSlots;
                   });

  uint64_t AssignedGateways = 0;
  for (const PendingGateway &P : Pending) {
    Trampoline &T = Ctx.OutTrampolines[P.TrampolineIndex];
    Expected<std::optional<EncodedSetPcGateway>> GatewayOrErr =
        findNearestSetPcGateway(Gateways, Ctx.LS, T.OriginalOffset,
                                P.TargetOffset, T.LongBranchSgprBase);
    if (!GatewayOrErr) {
      log() << "hotswap: error: failed to plan gateway for far site 0x"
            << utohexstr(T.OriginalOffset) << ": "
            << toString(GatewayOrErr.takeError()) << "\n";
      return false;
    }
    std::optional<EncodedSetPcGateway> Gateway = std::move(*GatewayOrErr);
    if (!Gateway) {
      log() << "hotswap: error: no safe short-branch gateway for far site 0x"
            << utohexstr(T.OriginalOffset) << " (" << P.InitialCandidateSlots
            << " initial candidate slot(s))\n";
      return false;
    }
    T.HasForwardGateway = true;
    T.ForwardGatewayOffset = Gateway->Sled->WritePos;
    T.ForwardGatewayBytes = std::move(Gateway->Bytes);
    Gateway->Sled->WritePos += T.ForwardGatewayBytes.size();
    ++AssignedGateways;
  }
  if (!Pending.empty())
    log() << "hotswap: assigned " << AssignedGateways
          << " SCC-neutral forward gateway(s)\n";
  if (BranchIslandChains != 0)
    log() << "hotswap: assigned " << BranchIslandChains
          << " forward s_branch island chain(s)\n";

  for (Trampoline &T : Ctx.OutTrampolines) {
    if (T.HasForwardGateway) {
      if (T.ForwardGatewayOffset > Ctx.TextSize ||
          T.ForwardGatewayBytes.size() >
              Ctx.TextSize - T.ForwardGatewayOffset) {
        log() << "hotswap: error: forward gateway at 0x"
              << utohexstr(T.ForwardGatewayOffset) << " extends past .text.\n";
        return false;
      }
      std::memcpy(Ctx.Text + T.ForwardGatewayOffset,
                  T.ForwardGatewayBytes.data(), T.ForwardGatewayBytes.size());
    }
    for (size_t I = 0; I != T.ForwardBranchIslands.size(); ++I) {
      uint64_t From = T.ForwardBranchIslands[I];
      uint64_t To = I + 1 == T.ForwardBranchIslands.size()
                        ? T.ForwardBranchTargetOffset
                        : T.ForwardBranchIslands[I + 1];
      SmallVector<uint8_t> Branch = Ctx.LS.encodeSBranch(From, To);
      if (Branch.size() != MinInstSize) {
        log() << "hotswap: error: failed to encode forward branch island at "
                 "0x"
              << utohexstr(From) << "\n";
        return false;
      }
      DenseMap<uint64_t, size_t>::const_iterator Owner =
          PoolIslandOwners.find(From);
      if (Owner != PoolIslandOwners.end()) {
        Trampoline &OwnerT = Ctx.OutTrampolines[Owner->second];
        std::memcpy(OwnerT.Bytes.data() + OwnerT.Bytes.size() -
                        PoolBranchIslandBytes,
                    Branch.data(), Branch.size());
      } else {
        if (From > Ctx.TextSize || Branch.size() > Ctx.TextSize - From) {
          log() << "hotswap: error: forward branch island at 0x"
                << utohexstr(From) << " is outside .text and trampoline pool\n";
          return false;
        }
        std::memcpy(Ctx.Text + From, Branch.data(), Branch.size());
      }
    }
  }
  return true;
}

/// Emit \p Replacement for the instruction at [\p InstOffset,
/// \p InstOffset + \p InstSize). Prefers an in-place NOP-sled rewrite when a
/// reachable sled with sufficient headroom exists; otherwise falls back to a
/// deferred trampoline.
[[nodiscard]] bool emitReplacementCode(PatchContext &Ctx, uint64_t InstOffset,
                                       uint32_t InstSize,
                                       ArrayRef<uint8_t> Replacement) {
  std::optional<uint64_t> ReturnTo = checkedAddUint64(
      InstOffset, InstSize, "replacement trampoline return target");
  std::optional<uint64_t> PoolReturnFrom =
      checkedAddUint64(Ctx.PoolBaseOffset, Replacement.size(),
                       "replacement trampoline return slot");
  if (!ReturnTo || !PoolReturnFrom)
    return false;

  // When the pool base is already out of short-branch reach, defer every site
  // to the global trampoline pass. That pass can coalesce adjacent patches
  // before allocating gateways; consuming NOP padding greedily here can strand
  // a later small or clause/delay-constrained source window.
  bool PoolBaseFar = !isSBranchReachable(InstOffset, Ctx.PoolBaseOffset) ||
                     !isSBranchReachable(*PoolReturnFrom, *ReturnTo);
  if (!PoolBaseFar && !Ctx.DirectControlFlow.HasUnresolvedTargets) {
    // findNearestSled enforces sled headroom. emitToNopSled still validates
    // exact branch reachability because branch-back distance includes the
    // replacement size, not just the original instruction offset.
    uint64_t Needed = Replacement.size() + MinInstSize;
    if (NopSled *Sled = findNearestSled(Ctx.NopSleds, InstOffset, Needed)) {
      if (emitToNopSled(Ctx, *Sled, InstOffset, InstSize, Replacement))
        return true;
      log() << "hotswap: emitReplacementCode: NOP sled at offset 0x"
            << utohexstr(Sled->WritePos)
            << " is not branch-reachable after assembly; using trampoline.\n";
    }
  }
  return emitToTrampoline(Ctx, InstOffset, InstSize, Replacement);
}

// -- applyGfx1250B0toA0Rules --------------------------------------------------

/// Per-instruction patch-pass trampoline: invokes \p Fn with (\p Ctx,
/// \p Idx) if it is non-null, or returns 0 otherwise. nullptr means
/// the corresponding pass family has no implementation linked in,
/// which the dispatcher treats as a no-op slot. std::nullopt means the
/// pass found a required patch failure after logging a specific reason.
static std::optional<uint32_t> runPerInstPass(uint32_t (*Fn)(PatchContext &,
                                                             size_t),
                                              PatchContext &Ctx, size_t Idx) {
  if (!Fn)
    return 0;

  uint32_t PatchCount = Fn(Ctx, Idx);
  if (Ctx.RequiredPatchFailed)
    return std::nullopt;
  return PatchCount;
}

/// Main per-instruction dispatcher for the GFX1250 B0-to-A0 rewrite.
/// Builds the NOP sled map, CFG, and VGPR liveness for the decoded stream,
/// then walks each decoded instruction and runs the patch passes in order
/// (in-place -> trampoline -> WMMA split -> scratch). Each pass gets a
/// chance to claim the instruction; first non-zero return wins. Also runs
/// the whole-function WMMA-hazard pass after the per-instruction loop and
/// records per-kernel stats via ElfView::updateKernelDescriptorVgprCount.
/// Returns the total number of applied patches across all passes.
static std::optional<uint32_t> applyGfx1250B0toA0Rules(
    std::vector<InternalDecodedInst> &Decoded, uint8_t *Text, uint64_t TextSize,
    const LLVMState &LS, std::vector<Trampoline> &OutTrampolines, ElfView &Elf,
    std::vector<ScratchPatchInfo> &OutScratchPatches,
    const RewriteConfig &Config, bool &OutRequiredPatchApplied,
    HotswapProfile &Profile) {
  uint32_t Patched = 0;

  HotswapProfile::Scope SledScope = Profile.time(HotswapMetric::NopSledScan);
  std::vector<NopSled> Sleds = buildNopSledMap(Decoded, LS, Elf);
  SledScope.finish();

  std::optional<DeclaredTextEntryInfo> DeclaredEntries =
      collectDeclaredTextEntries(Elf);
  if (!DeclaredEntries)
    return std::nullopt;
  std::vector<ElfView::FunctionTextRange> FunctionRanges =
      Elf.functionTextRanges();
  std::optional<DirectControlFlowInfo> ControlFlow = collectDirectBranchTargets(
      Decoded, LS, Elf.textAddr(), Elf.textSize(), DeclaredEntries->Entries,
      FunctionRanges, DeclaredEntries->ExternalEntries);
  if (!ControlFlow)
    return std::nullopt;
  if (ControlFlow->HasUnresolvedTargets) {
    log() << "hotswap: unresolved control-flow target disables NOP-sled "
             "emission, trampoline coalescing, source relocation, and .text "
             "gateways\n";
    Sleds.clear();
  } else {
    truncateNopSledsAtDirectTargets(Sleds, ControlFlow->Targets);
  }

  HotswapProfile::Scope CfgScope = Profile.time(HotswapMetric::CfgBuild);
  CFG Cfg = buildCfg(Decoded, *LS.MCII);
  CfgScope.finish();

  HotswapProfile::Scope LiveScope = Profile.time(HotswapMetric::Liveness);
  LivenessInfo Liveness =
      computeLiveness(Decoded, Cfg, *LS.MCII, *LS.MRI, Config.MaxVgprs);
  LiveScope.finish();

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
  // Pool base as a .text-relative offset for trampoline branch math. The pool
  // is always >= textAddr(); checkedSubUint64 guards a malformed object.
  std::optional<uint64_t> PoolVAddr = Elf.trampolinePoolVAddr();
  if (!PoolVAddr)
    return std::nullopt;
  std::optional<uint64_t> PoolBaseOffset = checkedSubUint64(
      *PoolVAddr, Elf.textAddr(), "trampoline pool base offset");
  if (!PoolBaseOffset)
    return std::nullopt;
  PatchContext Ctx{
      Config,      Decoded,           Text,         TextSize, *PoolBaseOffset,
      LS,          OutTrampolines,    Sleds,        Elf,      Liveness,
      KernelStats, OutScratchPatches, *ControlFlow, Profile};

  const HotswapPatchVTable &VT = getHotswapPatchVTable();

  // Skip undecoded slots produced by the decoder for bytes it could not
  // classify as a valid instruction; the dispatcher has nothing to match
  // against on these and we must not invoke the patch passes for them.
  constexpr StringLiteral UnknownMnemonic = "<unknown>";
  using PerInstPatchFn = uint32_t (*)(PatchContext &, size_t);
  // A pass plus its metric; time/patches are summed locally and flushed once
  // after the loop (see HotswapProfile::add).
  struct TimedPass {
    PerInstPatchFn Fn;
    HotswapMetric Metric;
    uint64_t Nanos = 0;
    uint64_t Patches = 0;
  };
  SmallVector<TimedPass, 5> Passes;
  if (Config.RunB0A0Patches) {
    Passes.push_back({VT.applyInPlacePatches, HotswapMetric::InPlace});
    Passes.push_back({VT.applyTrampolinePatches, HotswapMetric::Trampoline});
    Passes.push_back({VT.applyWmmaSplitPatches, HotswapMetric::WmmaSplit});
    Passes.push_back({VT.applyScratchPatches, HotswapMetric::ScratchFp8});
    Passes.push_back({VT.applyWmmaScale16Patches, HotswapMetric::WmmaScale16});
  } else {
    Passes.push_back({VT.applyTrampolinePatches, HotswapMetric::Trampoline});
  }

  const bool Prof = Ctx.Profile.enabled();

  for (size_t Idx = 0, E = Decoded.size(); Idx < E; ++Idx) {
    const InternalDecodedInst &DI = Decoded[Idx];
    if (DI.Mnemonic == UnknownMnemonic)
      continue;

    for (TimedPass &Pass : Passes) {
      const uint64_t T0 = Prof ? profNowNs() : 0;
      std::optional<uint32_t> P = runPerInstPass(Pass.Fn, Ctx, Idx);
      if (Prof) {
        Pass.Nanos += profNowNs() - T0;
        Pass.Patches += P.value_or(0);
      }
      if (!P)
        return std::nullopt;
      if (*P == 0)
        continue;
      Patched += *P;
      break;
    }
  }

  if (Prof)
    for (const TimedPass &Pass : Passes)
      Ctx.Profile.add(Pass.Metric, Pass.Nanos, Pass.Patches);

  // Whole-kernel passes below run after per-instruction patches. Earlier
  // passes may have modified Text bytes, but the Decoded stream still holds
  // the original MCInst/Mnemonic/Offset entries. This is safe because:
  //  - In-place patches only change opcodes within the same encoding size,
  //    preserving instruction boundaries and offsets.
  //  - Trampoline patches replace the original instruction with a branch
  //    (same size), so the Decoded entry's Offset still points at the
  //    branch site; the WMMA classifier and VOP3PX2 mnemonic match won't
  //    treat a branch as WMMA/VALU/VOP3PX2.
  // If a future patch family changes instruction boundaries, the Decoded
  // stream must be rebuilt before these passes run.
  if (Config.RunB0A0Patches && VT.applyWmmaHazardPatch) {
    HotswapProfile::Scope HazardScope =
        Ctx.Profile.time(HotswapMetric::WmmaHazard);
    const uint32_t P = VT.applyWmmaHazardPatch(Ctx);
    HazardScope.addPatches(P);
    HazardScope.finish();
    Patched += P;
  }
  if (Config.RunB0A0Patches && VT.applyVop3px2Src2Fix) {
    HotswapProfile::Scope Vop3Scope =
        Ctx.Profile.time(HotswapMetric::Vop3px2Src2);
    const uint32_t P = VT.applyVop3px2Src2Fix(Ctx);
    Vop3Scope.addPatches(P);
    Vop3Scope.finish();
    Patched += P;
  }

  if (!OutTrampolines.empty()) {
    if (!ControlFlow->HasUnresolvedTargets) {
      mergeAdjacentLongTrampolines(OutTrampolines, ControlFlow->Targets);
      expandStraightLineTrampolines(Ctx, ControlFlow->Targets);
      mergeAdjacentLongTrampolines(OutTrampolines, ControlFlow->Targets);
    }
    appendPoolBranchIslands(OutTrampolines);
    if (!assignLongBranchGateways(Ctx, ControlFlow->Targets,
                                  !ControlFlow->HasUnresolvedTargets))
      return std::nullopt;
  }

  struct ResourceCounts {
    unsigned Vgprs;
    unsigned Sgprs;
  };
  StringMap<ResourceCounts> CountsBefore;
  StringMap<unsigned> VgprGranules;
  StringMap<unsigned> RequiredVgprCounts;
  StringMap<unsigned> RequiredSgprCounts;
  for (const StringMapEntry<KernelPatchStats> &KV : KernelStats) {
    StringRef KName = KV.first();
    const KernelPatchStats &Stats = KV.second;
    if (KName.empty())
      continue;
    unsigned VgprGranule = getKernelVgprGranuleSize(Ctx, KName);
    VgprGranules.try_emplace(KName, VgprGranule);
    std::optional<unsigned> VgprsBefore =
        Elf.getKernelVgprCount(KName, VgprGranule);
    std::optional<unsigned> SgprsBefore = Elf.getKernelSgprCount(KName);
    CountsBefore.try_emplace(KName, ResourceCounts{VgprsBefore.value_or(0),
                                                   SgprsBefore.value_or(0)});
    if (Stats.ExtraVgprs > 0) {
      // Every current VGPR-growing patch preflights before emitting bytes.
      // Keep this required-policy check as a fail-safe so a future path cannot
      // silently emit a kernel that no longer admits one maximum workgroup.
      if (checkKernelVgprBump(Ctx, KName, Stats.ExtraVgprs,
                              PatchRequirement::Required) !=
          VgprBumpDecision::Apply)
        return std::nullopt;
      if (!VgprsBefore) {
        log() << "hotswap: error: failed to read VGPR count for kernel "
              << KName << "\n";
        return std::nullopt;
      }
      if (Stats.ExtraVgprs >
          std::numeric_limits<unsigned>::max() - *VgprsBefore) {
        log() << "hotswap: error: VGPR count for kernel " << KName
              << " overflows unsigned after hotswap scratch allocation\n";
        return std::nullopt;
      }
      RequiredVgprCounts.try_emplace(KName, *VgprsBefore + Stats.ExtraVgprs);
    }
    if (Stats.ExtraSgprs > 0) {
      if (!SgprsBefore) {
        log() << "hotswap: error: failed to read SGPR count for kernel "
              << KName << "\n";
        return std::nullopt;
      }
      if (Stats.ExtraSgprs >
          std::numeric_limits<unsigned>::max() - *SgprsBefore) {
        log() << "hotswap: error: SGPR count for kernel " << KName
              << " overflows unsigned after hotswap scratch allocation\n";
        return std::nullopt;
      }
      unsigned RequiredSgprs = *SgprsBefore + Stats.ExtraSgprs;
      RequiredSgprCounts.try_emplace(KName, RequiredSgprs);
    }
  }

  if (!Elf.updateKernelMetadataVgprCounts(RequiredVgprCounts)) {
    log() << "hotswap: error: failed to update kernel VGPR metadata\n";
    return std::nullopt;
  }
  if (!Elf.updateKernelMetadataSgprCounts(RequiredSgprCounts)) {
    log() << "hotswap: error: failed to update kernel SGPR metadata\n";
    return std::nullopt;
  }
  for (const StringMapEntry<unsigned> &Required : RequiredVgprCounts) {
    StringMap<unsigned>::const_iterator Granule =
        VgprGranules.find(Required.first());
    if (Granule == VgprGranules.end()) {
      log() << "hotswap: error: missing VGPR granule for kernel "
            << Required.first() << "\n";
      return std::nullopt;
    }
    if (!Elf.updateKernelDescriptorVgprCount(Required.first(), Required.second,
                                             Granule->second)) {
      log() << "hotswap: error: failed to update VGPR descriptor count for "
            << Required.first() << "\n";
      return std::nullopt;
    }
  }

  for (const StringMapEntry<KernelPatchStats> &KV : KernelStats) {
    StringRef KName = KV.first();
    const KernelPatchStats &Stats = KV.second;
    if (KName.empty())
      continue;
    StringMap<ResourceCounts>::const_iterator Before = CountsBefore.find(KName);
    if (Before == CountsBefore.end()) {
      log() << "hotswap: error: missing cached resource counts for kernel "
            << KName << "\n";
      return std::nullopt;
    }
    StringMap<unsigned>::const_iterator Granule = VgprGranules.find(KName);
    if (Granule == VgprGranules.end()) {
      log() << "hotswap: error: missing VGPR granule for kernel " << KName
            << "\n";
      return std::nullopt;
    }
    std::optional<unsigned> VgprsAfter =
        Elf.getKernelVgprCount(KName, Granule->second);
    std::optional<unsigned> SgprsAfter = Elf.getKernelSgprCount(KName);
    log() << "hotswap: liveness: kernel " << KName
          << ": vgprs_before=" << Before->second.Vgprs
          << ", vgprs_after=" << VgprsAfter.value_or(0)
          << ", sgprs_before=" << Before->second.Sgprs
          << ", sgprs_after=" << SgprsAfter.value_or(0)
          << ", scratch_reused=" << Stats.ScratchReused
          << ", scratch_above_kd=" << Stats.ScratchAboveKd << "\n";
  }
  OutRequiredPatchApplied = Ctx.RequiredPatchApplied;
  return Patched;
}

// -- retargetCodeObject helpers -------------------------------------------

/// Finalize the deferred trampolines produced by emitToTrampoline: resolves
/// the branch-back at the tail of each trampoline to land on the next
/// instruction after the original site, writes the branch-forward + s_nop
/// padding at the original .text slot, and reports per-trampoline encoding
/// failures through log(). Runs after all patch passes finish so the
/// post-.text layout of trampolines is known. Returns false if any
/// trampoline could not be fixed up.
[[nodiscard]] static bool
fixupTrampolineBranches(std::vector<Trampoline> &Trampolines, uint8_t *Text,
                        uint64_t PoolBaseOffset, const LLVMState &LS) {
  // Fail-fast on the first encoding error: the position of later
  // trampolines depends on earlier ones, so a single bad branch would
  // cascade into incorrect layout. A single failure invalidates the whole
  // rewrite, so there is nothing useful to recover beyond it.
  //
  // Offsets are .text-relative; the pool begins at PoolBaseOffset
  // (trampolinePoolVAddr() - textAddr()), which can be far past .text.
  uint64_t TrampOffset = PoolBaseOffset;
  for (Trampoline &T : Trampolines) {
    uint64_t TP = TrampOffset;
    std::optional<uint64_t> NextTrampOffset = checkedAddUint64(
        TrampOffset, T.Bytes.size(), "trampoline fixup layout");
    if (!NextTrampOffset)
      return false;
    TrampOffset = *NextTrampOffset;

    if (T.Long && !T.UsesSetPCBack) {
      log() << "hotswap: error: far trampoline lacks safe set-PC return at 0x"
            << utohexstr(T.OriginalOffset) << "\n";
      return false;
    }
    const uint32_t BackReserve =
        T.UsesSetPCBack ? SetPcReturnReserveBytes : MinInstSize;
    const uint32_t TrailingIsland =
        T.HasPoolBranchIsland ? PoolBranchIslandBytes : 0;
    if (T.Bytes.size() < BackReserve + TrailingIsland) {
      log() << "hotswap: error: trampoline return reservation is truncated at "
               "0x"
            << utohexstr(T.OriginalOffset) << "\n";
      return false;
    }
    const uint64_t BackSlot = TrampOffset - TrailingIsland - BackReserve;
    const size_t BackOffset = T.Bytes.size() - TrailingIsland - BackReserve;
    std::optional<uint64_t> ReturnTo = checkedAddUint64(
        T.OriginalOffset, T.OriginalSize, "trampoline return target");
    if (!ReturnTo)
      return false;

    std::optional<SmallVector<uint8_t>> BrBack;
    if (T.UsesSetPCBack) {
      BrBack =
          encodeSetPCLongBranch(LS, BackSlot, *ReturnTo, T.LongBranchSgprBase);
    } else {
      SmallVector<uint8_t> ShortBranch = LS.encodeSBranch(BackSlot, *ReturnTo);
      if (!ShortBranch.empty())
        BrBack = std::move(ShortBranch);
    }
    if (!BrBack || BrBack->size() > BackReserve) {
      log() << "hotswap: error: trampoline branch-back encoding failed at 0x"
            << utohexstr(T.OriginalOffset) << (T.Long ? " (long)\n" : "\n");
      return false;
    }
    std::memcpy(T.Bytes.data() + BackOffset, BrBack->data(), BrBack->size());
    for (uint32_t I = BrBack->size(); I + MinInstSize <= BackReserve;
         I += MinInstSize)
      std::memcpy(T.Bytes.data() + BackOffset + I, LS.SNopBytes.data(),
                  MinInstSize);

    SmallVector<uint8_t> BrFwd;
    if (T.Long) {
      if (T.UsesShortBranchForward) {
        BrFwd = LS.encodeSBranch(T.OriginalOffset, TP);
      } else if (!T.ForwardBranchIslands.empty()) {
        BrFwd =
            LS.encodeSBranch(T.OriginalOffset, T.ForwardBranchIslands.front());
      } else if (T.UsesDirectSetPCForward) {
        BrFwd = T.DirectSetPCForwardBytes;
      } else if (T.HasForwardGateway) {
        BrFwd = LS.encodeSBranch(T.OriginalOffset, T.ForwardGatewayOffset);
      } else {
        log() << "hotswap: error: far trampoline has no forward gateway at 0x"
              << utohexstr(T.OriginalOffset) << "\n";
        return false;
      }
    } else {
      BrFwd = LS.encodeSBranch(T.OriginalOffset, TP);
    }
    if (BrFwd.empty() || BrFwd.size() > T.OriginalSize) {
      log() << "hotswap: error: trampoline branch-fwd encoding failed at 0x"
            << utohexstr(T.OriginalOffset) << (T.Long ? " (long)\n" : "\n");
      return false;
    }
    std::memcpy(Text + T.OriginalOffset, BrFwd.data(), BrFwd.size());
    // Pad the tail of the replaced slot with cached s_nop bytes.
    for (uint32_t I = BrFwd.size(); I + MinInstSize <= T.OriginalSize;
         I += MinInstSize)
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
                               const ElfView &Elf, size_t GrowthTotal) {
  uint8_t *Data = reinterpret_cast<uint8_t *>(ElfBuf.getBufferStart());
  size_t Size = ElfBuf.getBufferSize();
  if (!addTrampolineSymbols(ElfBuf, Trampolines, Elf.textSize(),
                            Elf.textSectionIndex()))
    log() << "hotswap: error: addTrampolineSymbols failed\n";
  patchDebugRanges(Data, Size, Elf.textAddr(), Elf.textSize(), GrowthTotal);
  patchDebugInfo(Data, Size, Elf.textAddr(), Elf.textSize(), GrowthTotal);
  patchDebugFrame(Data, Size, Elf.textAddr(), Elf.textSize(), GrowthTotal);
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

static std::unique_ptr<WritableMemoryBuffer>
copyOutputBuffer(const void *Data, size_t Size, StringRef CopyKind) {
  std::unique_ptr<WritableMemoryBuffer> Result =
      WritableMemoryBuffer::getNewUninitMemBuffer(Size);
  if (!Result) {
    log() << "hotswap: error: retargetCodeObject: "
          << "getNewUninitMemBuffer(" << Size
          << ") failed (out of memory) for the " << CopyKind
          << " output copy.\n";
    return nullptr;
  }

  std::memcpy(Result->getBufferStart(), Data, Size);
  return Result;
}

// -- retargetCodeObject -------------------------------------------------------

static amd_comgr_status_t retargetCodeObjectImpl(
    const void *ElfData, size_t ElfSize, const TargetIdentifier &TargetIdent,
    const Gfx1250RewriteOptions &Options, std::unique_ptr<MemoryBuffer> &Out,
    bool AllowTextDisplacement, HotswapProfile &Profile) {
  // The dispatcher fetches the patch vtable lazily via
  // getHotswapPatchVTable() inside applyGfx1250B0toA0Rules; the singleton's
  // initializer binds every register*Patch slot on first access, so no
  // explicit install step is needed here.

  const bool RunInstructionPatches =
      Options.RunB0A0Patches ||
      Options.MaskPolicy != MaskWorkaroundPolicy::None;
  const bool Prof = Profile.enabled();

  // Take a working copy so the input is preserved and we have a mutable
  // buffer to parse / patch.
  uint64_t InputCopyT0 = Prof ? profNowNs() : 0;
  std::vector<uint8_t> Buf(static_cast<const uint8_t *>(ElfData),
                           static_cast<const uint8_t *>(ElfData) + ElfSize);
  if (Prof)
    Profile.add(HotswapMetric::InputCopy, profNowNs() - InputCopyT0, 0);

  uint64_t ParseT0 = Prof ? profNowNs() : 0;
  Expected<ElfView> ViewOrErr = ElfView::create(Buf.data(), Buf.size());
  if (!ViewOrErr) {
    log() << "hotswap: error: retargetCodeObject: input is not a "
          << "parseable ELF64 (" << toString(ViewOrErr.takeError()) << ").\n";
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }
  if (ViewOrErr->textSize() == 0) {
    log() << "hotswap: error: retargetCodeObject: input ELF has empty "
          << ".text section; nothing to rewrite.\n";
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }
  ElfView &Elf = *ViewOrErr;
  if (Prof)
    Profile.add(HotswapMetric::ElfParse, profNowNs() - ParseT0, 0);

  // The CPU name and s_nop padding bytes are the only rewrite state the fast
  // path needs; both are also carried by LLVMState on the MC path. Holding them
  // as standalone locals lets the shared tail work off them regardless of which
  // path ran, so the fast path never builds an LLVMState.
  const StringRef TargetCpu = TargetIdent.Processor;
  static constexpr uint8_t SNop[4] = {0x00, 0x00, 0x80, 0xbf};
  SmallVector<uint8_t, 4> SNopBytes(SNop, SNop + sizeof(SNop));

  // B0->B0 entry-only fast path: no instruction patches means no .text decode,
  // so skip the whole LLVM MC layer and emit entry stubs from a pre-encoded
  // byte template. UseB0B0EntryFastPath is decided by the caller from the
  // source/target stepping; the template bytes and the HWSD workaround are
  // gfx1250-specific, which that flag already accounts for.
  const bool UseFastAppend = Options.RunEntryTrampolines &&
                             !RunInstructionPatches &&
                             Options.UseB0B0EntryFastPath;

  // The MC layer (disassembler, encoder, register info) is only initialized on
  // the non-fast path. The fast path leaves LS default-constructed and unused:
  // it works entirely off TargetCpu / SNopBytes above, and every LS access
  // below is guarded by a condition that is false on the fast path.
  LLVMState LS;
  if (UseFastAppend) {
    log() << "hotswap: entry trampolines: B0->B0 fast path (no MC/.text "
             "disassembly)\n";
  } else {
    uint64_t InitT0 = Prof ? profNowNs() : 0;
    LS = initLLVM(TargetIdent);
    if (Prof)
      Profile.add(HotswapMetric::InitLLVM, profNowNs() - InitT0, 0);
    if (!LS.Valid) {
      log() << "hotswap: error: retargetCodeObject: initLLVM failed "
            << "for CPU '" << TargetIdent.Processor << "'; aborting rewrite.\n";
      return AMD_COMGR_STATUS_ERROR;
    }
  }

  // Direct displacement is an entry-workaround optimization only. Apply the
  // entry prefixes before ordinary instruction rewriting so the existing
  // NOP-sled/trampoline planner sees the final instruction offsets. If the ELF
  // cannot be displaced safely, continue from the pristine working copy and
  // append the established entry stubs below.
  if (Options.RunEntryTrampolines && AllowTextDisplacement && !UseFastAppend) {
    std::vector<DisplacementEdit> EntryDisplacements;
    std::optional<uint32_t> EntryCount =
        collectKernelEntryDisplacements(Elf, LS, EntryDisplacements);
    if (!EntryCount)
      return AMD_COMGR_STATUS_ERROR;

    if (!EntryDisplacements.empty()) {
      Expected<std::unique_ptr<WritableMemoryBuffer>> DisplacedOrErr =
          tryApplyTextDisplacementToNewBuffer(Elf, LS, EntryDisplacements);
      if (DisplacedOrErr) {
        std::unique_ptr<WritableMemoryBuffer> Displaced =
            std::move(*DisplacedOrErr);
        if (!RunInstructionPatches) {
          Out = std::move(Displaced);
          return AMD_COMGR_STATUS_SUCCESS;
        }

        Gfx1250RewriteOptions RemainingOptions = Options;
        RemainingOptions.RunEntryTrampolines = false;
        RemainingOptions.UseB0B0EntryFastPath = false;
        return retargetCodeObjectImpl(Displaced->getBufferStart(),
                                      Displaced->getBufferSize(), TargetIdent,
                                      RemainingOptions, Out,
                                      /*AllowTextDisplacement=*/false, Profile);
      }

      log() << "hotswap: entry displacement unavailable: "
            << toString(DisplacedOrErr.takeError())
            << "; using appended entry stubs\n";
    }
  }

  RewriteConfig Config = makeGfx1250B0A0Config();
  Config.RunB0A0Patches = Options.RunB0A0Patches;
  Config.MaskPolicy = Options.MaskPolicy;

  uint8_t *Text = Elf.textData();
  uint64_t Count = 0;
  std::vector<Trampoline> Deferred;
  std::vector<ScratchPatchInfo> ScratchPatches;
  bool RequiredPatchApplied = false;
  if (RunInstructionPatches) {
    std::vector<InternalDecodedInst> Decoded;
    uint64_t DecodeT0 = Prof ? profNowNs() : 0;
    bool DecodedOk = decodeTextSection(Text, Elf.textSize(), LS, Decoded);
    if (Prof)
      Profile.add(HotswapMetric::Decode, profNowNs() - DecodeT0, 0);
    if (!DecodedOk) {
      log() << "hotswap: error: retargetCodeObject: decodeTextSection "
            << "failed on .text (" << Elf.textSize() << " bytes).\n";
      return AMD_COMGR_STATUS_ERROR;
    }

    uint64_t DispatchT0 = Prof ? profNowNs() : 0;
    std::optional<uint32_t> Patched = applyGfx1250B0toA0Rules(
        Decoded, Text, Elf.textSize(), LS, Deferred, Elf, ScratchPatches,
        Config, RequiredPatchApplied, Profile);
    if (Prof)
      Profile.add(HotswapMetric::B0A0Dispatch, profNowNs() - DispatchT0, 0);
    if (!Patched)
      return AMD_COMGR_STATUS_ERROR;
    Count = *Patched;
    log() << "hotswap: applied " << Count << " instruction patches\n";
  } else {
    log() << "hotswap: instruction patches disabled for this rewrite\n";
  }

  // gfx1250 revision is recorded per kernel in the AMDGPU metadata note.
  // Running a B0 object on A0 requires retagging that metadata even when no
  // machine instruction needed rewriting.
  if (Options.RunB0A0Patches && !Elf.updateGfx1250RevisionMetadata("A0"))
    return AMD_COMGR_STATUS_ERROR;

  std::unique_ptr<WritableMemoryBuffer> Result;
  uint64_t PoolT0 = Prof ? profNowNs() : 0;
  std::vector<Trampoline> Growth = Deferred;
  // The appended pool's fresh virtual address is the single reference point for
  // all trampoline branch/stub targets (growWithTrampolines places it there).
  std::optional<uint64_t> PoolVAddrOr = Elf.trampolinePoolVAddr();
  if (!PoolVAddrOr) {
    log() << "hotswap: error: retargetCodeObject: could not compute trampoline "
          << "pool virtual address.\n";
    return AMD_COMGR_STATUS_ERROR;
  }
  const uint64_t PoolVAddr = *PoolVAddrOr;
  // Pool is always >= textAddr(); checkedSubUint64 guards a malformed object.
  std::optional<uint64_t> PoolBaseOffsetOr = checkedSubUint64(
      PoolVAddr, Elf.textAddr(), "trampoline pool base offset");
  if (!PoolBaseOffsetOr)
    return AMD_COMGR_STATUS_ERROR;
  const uint64_t PoolBaseOffset = *PoolBaseOffsetOr;
  if (Prof)
    Profile.add(HotswapMetric::PoolSetup, profNowNs() - PoolT0, 0);
  if (!Deferred.empty()) {
    uint64_t FixupT0 = Prof ? profNowNs() : 0;
    bool FixupOk = fixupTrampolineBranches(Deferred, Text, PoolBaseOffset, LS);
    if (Prof)
      Profile.add(HotswapMetric::FixupTrampolines, profNowNs() - FixupT0, 0);
    if (!FixupOk) {
      if (RequiredPatchApplied) {
        log() << "hotswap: error: required patch trampoline branch fixup "
                 "failed; refusing to return the original unsafe code "
                 "object\n";
        return AMD_COMGR_STATUS_ERROR;
      }
      // A trampoline branch could not be encoded, so the local `Buf` copy
      // is half-redirected; shipping it would run corrupted code. Fall back
      // to the pristine input object (`ElfData`, untouched) so the loader
      // runs the original unpatched code instead.
      log() << "hotswap: error: some trampolines could not be fixed up; "
            << "falling back to the original (unpatched) code object\n";
      std::unique_ptr<WritableMemoryBuffer> Orig =
          WritableMemoryBuffer::getNewUninitMemBuffer(ElfSize);
      if (!Orig) {
        log() << "hotswap: error: retargetCodeObject: "
              << "getNewUninitMemBuffer(" << ElfSize
              << ") failed (out of memory) for the fallback copy.\n";
        return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
      }
      std::memcpy(Orig->getBufferStart(), ElfData, ElfSize);
      Out = std::move(Orig);
      // SUCCESS here is misleading the returned buffer is the
      // *unpatched* original, so callers cannot tell "rewrote successfully"
      // from "declined and fell back". The status vocabulary needs a distinct
      // "no-op / not-applied" code.
      return AMD_COMGR_STATUS_SUCCESS;
    }
    Growth = Deferred;
  }

  std::vector<KernelEntryTrampolineFixup> EntryFixups;
  if (Options.RunEntryTrampolines) {
    uint64_t EntryT0 = Prof ? profNowNs() : 0;
    std::optional<uint32_t> EntryCount =
        UseFastAppend
            ? appendKernelEntryTrampolinesFast(Elf, TargetCpu, Config.MaxSgprs,
                                               Growth, EntryFixups)
            : appendKernelEntryTrampolines(Elf, LS, Config.MaxSgprs, Growth,
                                           EntryFixups);
    if (Prof)
      Profile.add(HotswapMetric::EntryTrampolines, profNowNs() - EntryT0,
                  EntryCount.value_or(0));
    if (!EntryCount)
      return AMD_COMGR_STATUS_ERROR;
    Count += *EntryCount;
  } else {
    log() << "hotswap: kernel-entry trampolines disabled for this rewrite\n";
  }

  if (!Deferred.empty()) {
    uint64_t GuardT0 = Prof ? profNowNs() : 0;
    bool GuardOk = appendDeferredTrampolinePrefetchGuard(Elf, LS, Growth);
    if (Prof)
      Profile.add(HotswapMetric::PrefetchGuard, profNowNs() - GuardT0, 0);
    if (!GuardOk)
      return AMD_COMGR_STATUS_ERROR;
  }

  if (!Growth.empty()) {
    uint64_t GrowT0 = Prof ? profNowNs() : 0;
    Result = Elf.growWithTrampolines(Growth, SNopBytes);
    if (Prof)
      Profile.add(HotswapMetric::GrowElf, profNowNs() - GrowT0, 0);
    if (!Result) {
      log() << "hotswap: error: retargetCodeObject: "
            << "ElfView::growWithTrampolines returned null with "
            << Growth.size() << " trampolines queued.\n";
      return AMD_COMGR_STATUS_ERROR;
    }

    size_t GrowthTotal = 0;
    for (const Trampoline &T : Growth) {
      if (T.Bytes.size() > std::numeric_limits<size_t>::max() - GrowthTotal) {
        log() << "hotswap: error: retargetCodeObject: growth byte count "
              << "overflows size_t.\n";
        return AMD_COMGR_STATUS_ERROR;
      }
      GrowthTotal += T.Bytes.size();
    }
    uint64_t DbgT0 = Prof ? profNowNs() : 0;
    patchDebugSections(*Result, Deferred, Elf, GrowthTotal);
    if (Prof)
      Profile.add(HotswapMetric::DebugSections, profNowNs() - DbgT0, 0);

    uint64_t KdT0 = Prof ? profNowNs() : 0;
    bool KdOk = rewriteKernelEntryDescriptorOffsets(*Result, PoolVAddr,
                                                    TargetCpu, EntryFixups);
    if (Prof)
      Profile.add(HotswapMetric::KdRewrite, profNowNs() - KdT0, 0);
    if (!KdOk)
      return AMD_COMGR_STATUS_ERROR;

    // Give each appended entry stub a `<kernel>.stub` symbol so a dispatch
    // whose entry now points at the stub still resolves to a name (e.g. rocgdb
    // `info dispatches`). This grows only the non-alloc .symtab/.strtab and
    // returns a new buffer; failure is non-fatal (the rewritten code object is
    // still correct, just missing the debug-only symbol).
    //
    // FAST PATH: this .symtab/.strtab rebuild + full buffer copy scales with
    // kernel count and is pure overhead for a load-time-critical path (the ROCr
    // loader trampoline adds no such symbols). The symbols are only a debugging
    // aid, so the fast path skips them by default. Set
    // AMD_COMGR_HOTSWAP_ENTRY_STUB_SYMBOLS=1 to re-enable (e.g. for rocgdb).
    const bool AddStubSymbols =
        !UseFastAppend || env::shouldAddEntryTrampolineSymbols();
    if (!EntryFixups.empty() && AddStubSymbols) {
      uint64_t SymT0 = Prof ? profNowNs() : 0;
      std::unique_ptr<WritableMemoryBuffer> WithSyms =
          addKernelEntryTrampolineSymbols(*Result, Elf.textSectionIndex(),
                                          Elf.textAddr(), Elf.textSize(),
                                          EntryFixups);
      if (Prof)
        Profile.add(HotswapMetric::SymbolInsert, profNowNs() - SymT0, 0);
      if (WithSyms)
        Result = std::move(WithSyms);
    }
  } else {
    uint64_t OutCopyT0 = Prof ? profNowNs() : 0;
    Result = copyOutputBuffer(Buf.data(), ElfSize, "patched");
    if (Prof)
      Profile.add(HotswapMetric::OutputCopy, profNowNs() - OutCopyT0, 0);
    if (!Result)
      return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  if (!ScratchPatches.empty()) {
    uint64_t VerifyT0 = Prof ? profNowNs() : 0;
    runScratchVerification(*Result, LS, ScratchPatches, Config.MaxVgprs);
    if (Prof)
      Profile.add(HotswapMetric::ScratchVerify, profNowNs() - VerifyT0, 0);
  }

  Out = std::move(Result);
  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t retargetCodeObject(const void *ElfData, size_t ElfSize,
                                      const TargetIdentifier &TargetIdent,
                                      const Gfx1250RewriteOptions &Options,
                                      std::unique_ptr<MemoryBuffer> &Out) {
  const bool RunInstructionPatches =
      Options.RunB0A0Patches ||
      Options.MaskPolicy != MaskWorkaroundPolicy::None;
  if (!RunInstructionPatches && !Options.RunEntryTrampolines) {
    std::unique_ptr<WritableMemoryBuffer> Result =
        copyOutputBuffer(ElfData, ElfSize, "no-op");
    if (!Result)
      return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
    Out = std::move(Result);
    return AMD_COMGR_STATUS_SUCCESS;
  }

  // One profiling session per code object, merged into TimeStatistics when it
  // goes out of scope. Prof gates the manual per-phase clock reads.
  HotswapProfile Profile(hotswapProfilingEnabled());
  // RAII guard: records phase:rewrite_total on every return path.
  [[maybe_unused]] HotswapProfile::Scope TotalScope =
      Profile.time(HotswapMetric::RewriteTotal);

  return retargetCodeObjectImpl(ElfData, ElfSize, TargetIdent, Options, Out,
                                /*AllowTextDisplacement=*/true, Profile);
}

} // namespace hotswap
} // namespace COMGR
