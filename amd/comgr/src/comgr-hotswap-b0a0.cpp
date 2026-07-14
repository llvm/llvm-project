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

#include "comgr-hotswap-internal.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Compiler.h"

#include <algorithm>
#include <cassert>
#include <limits>

using namespace llvm;

namespace COMGR {
namespace hotswap {

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
// decode/encode helpers (getKernelVgprCount / updateKernelDescriptor) to
// interpret COMPUTE_PGM_RSRC1.GRANULATED_WORKITEM_VGPR_COUNT.
// GFX12 wave32: 106 user-addressable SGPRs (s0-s105); s106-s107 are VCC.
static constexpr unsigned Gfx1250MaxSgprs = 106;
static constexpr unsigned Gfx1250VgprGranuleSize = 16;

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
  std::vector<KernelDescriptorInfo> Descriptors = Elf.kernelDescriptors();
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
                                 uint64_t End,
                                 const ElfView::FunctionTextRange &Range) {
  if (End - Start >= MinNopSledSize)
    Sleds.push_back({Start, End, Start, Range.Begin, Range.End});
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
  return true;
}

// s_add_pc_i64 long branch from \p FromOffset to \p TargetOffset (.text
// offsets). It adds a signed literal to the next instruction's PC, so the
// offset is TargetOffset - (FromOffset + size); size is 8 (forward, 32-bit
// literal) or 12 (backward, 64-bit literal), resolved by trying each. Encoded
// via the MC assembler. Returns empty on failure.
//
// Precondition: callers only long-branch *far* sites, so the target is far
// enough that s_add_pc_i64 always needs a real (>= 32-bit) literal. A tiny
// offset would instead assemble to the 4-byte inline-constant form, match
// neither candidate size, and yield empty. That is asserted here (assert
// liberally); the empty return is kept as a release-build safety net so a
// stray near target degrades to the unpatched-object fallback rather than
// crashing the loader.
SmallVector<uint8_t> encodeLongBranch(const LLVMState &LS, uint64_t FromOffset,
                                      uint64_t TargetOffset) {
  [[maybe_unused]] const int64_t Distance =
      static_cast<int64_t>(TargetOffset) - static_cast<int64_t>(FromOffset);
  [[maybe_unused]] const int64_t MinLongDistance = LongBranchMaxBytes;
  assert((Distance > MinLongDistance || Distance < -MinLongDistance) &&
         "encodeLongBranch: target is not a far site; the long-branch path is "
         "only valid for offsets beyond an s_add_pc_i64 instruction's reach");

  for (uint32_t Size : {LongBranchFwdBytes, LongBranchMaxBytes}) {
    int64_t Off = static_cast<int64_t>(TargetOffset) -
                  static_cast<int64_t>(FromOffset + Size);
    SmallVector<uint8_t> Bytes =
        assembleSingleInst("s_add_pc_i64 " + std::to_string(Off), LS);
    if (Bytes.size() == Size)
      return Bytes;
  }
  return {};
}

std::optional<SmallVector<uint8_t>>
encodeSccNeutralLongBranch(const LLVMState &LS, uint64_t FromOffset,
                           uint64_t TargetOffset, unsigned SgprBase) {
  if ((SgprBase & 1u) != 0 ||
      SgprBase == std::numeric_limits<unsigned>::max()) {
    log() << "hotswap: error: SCC-neutral long branch requires an aligned "
             "SGPR pair, got s"
          << SgprBase << "\n";
    return std::nullopt;
  }

  const std::string Pair = "s[" + std::to_string(SgprBase) + ":" +
                           std::to_string(SgprBase + 1) + "]";
  SmallVector<uint8_t> GetPc = assembleSingleInst("s_get_pc_i64 " + Pair, LS);
  if (GetPc.empty()) {
    log() << "hotswap: error: SCC-neutral long branch failed to assemble "
             "s_get_pc_i64 for "
          << Pair << "\n";
    return std::nullopt;
  }

  std::optional<uint64_t> PcBase = checkedAddUint64(
      FromOffset, GetPc.size(), "SCC-neutral long branch PC base");
  if (!PcBase)
    return std::nullopt;

  // Unsigned subtraction intentionally materializes the two's-complement
  // delta for backward branches.
  const uint64_t Delta = TargetOffset - *PcBase;
  SmallVector<uint8_t> Add = assembleSingleInst(
      "s_add_nc_u64 " + Pair + ", " + Pair + ", 0x" + utohexstr(Delta), LS);
  if (Add.empty()) {
    log() << "hotswap: error: SCC-neutral long branch failed to assemble "
             "s_add_nc_u64 for "
          << Pair << "\n";
    return std::nullopt;
  }
  SmallVector<uint8_t> SetPc = assembleSingleInst("s_set_pc_i64 " + Pair, LS);
  if (SetPc.empty()) {
    log() << "hotswap: error: SCC-neutral long branch failed to assemble "
             "s_set_pc_i64 for "
          << Pair << "\n";
    return std::nullopt;
  }

  SmallVector<uint8_t> Bytes;
  Bytes.append(GetPc.begin(), GetPc.end());
  Bytes.append(Add.begin(), Add.end());
  Bytes.append(SetPc.begin(), SetPc.end());
  return Bytes;
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
  return Reg.isValid() && StringRef(LS.MRI->getName(Reg)).starts_with("VCC");
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

std::optional<SafeSgprScratchBlock>
findSafeSgprScratchBlock(const PatchContext &Ctx, uint64_t TextOffset,
                         unsigned Count, unsigned Alignment,
                         StringRef Context) {
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
  if (!ScanWholeObject) {
    for (const InternalDecodedInst &DI : Ctx.Decoded) {
      if (DI.Offset < FunctionRange->Begin || DI.Offset >= FunctionRange->End)
        continue;
      if (Ctx.LS.MIA && Ctx.LS.MIA->isCall(DI.Inst)) {
        ScanWholeObject = true;
        break;
      }
    }
  }

  bool UsesVcc = false;
  unsigned HighWatermark = 0;
  for (const InternalDecodedInst &DI : Ctx.Decoded) {
    if (!ScanWholeObject &&
        (DI.Offset < FunctionRange->Begin || DI.Offset >= FunctionRange->End))
      continue;
    UsesVcc |= instructionUsesVcc(Ctx.LS, DI);
    for (const MCOperand &Op : DI.Inst) {
      if (!Op.isReg() || !Op.getReg())
        continue;
      if (!updateNumberedSgprHighWatermark(*Ctx.LS.MRI, MCRegister(Op.getReg()),
                                           Ctx.Config.MaxSgprs, HighWatermark,
                                           Context))
        return std::nullopt;
    }

    const MCInstrDesc &Desc = Ctx.LS.MCII->get(DI.Inst.getOpcode());
    for (MCPhysReg Reg : Desc.implicit_uses())
      if (!updateNumberedSgprHighWatermark(*Ctx.LS.MRI, MCRegister(Reg),
                                           Ctx.Config.MaxSgprs, HighWatermark,
                                           Context))
        return std::nullopt;
    for (MCPhysReg Reg : Desc.implicit_defs())
      if (!updateNumberedSgprHighWatermark(*Ctx.LS.MRI, MCRegister(Reg),
                                           Ctx.Config.MaxSgprs, HighWatermark,
                                           Context))
        return std::nullopt;
  }

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
  std::vector<KernelDescriptorInfo> Descriptors = Ctx.Elf.kernelDescriptors();
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

static std::optional<SmallVector<uint8_t>>
buildSafeFarReturn(PatchContext &Ctx, uint64_t InstOffset, uint32_t InstSize,
                   uint64_t BackStart) {
  std::optional<SafeSgprScratchBlock> Scratch = findSafeSgprScratchBlock(
      Ctx, InstOffset, /*Count=*/2, /*Alignment=*/2, "safe far return");
  if (!Scratch)
    return std::nullopt;

  std::optional<uint64_t> ReturnTo =
      checkedAddUint64(InstOffset, InstSize, "safe far return target offset");
  if (!ReturnTo)
    return std::nullopt;
  std::optional<SmallVector<uint8_t>> Bytes =
      encodeSccNeutralLongBranch(Ctx.LS, BackStart, *ReturnTo, Scratch->Base);
  if (!Bytes)
    return std::nullopt;
  if (!commitSafeSgprScratchBlock(Ctx, InstOffset, *Scratch, "safe far return"))
    return std::nullopt;

  const std::string Pair = "s[" + std::to_string(Scratch->Base) + ":" +
                           std::to_string(Scratch->Base + 1) + "]";
  log() << "hotswap: safe far return at 0x" << utohexstr(InstOffset) << " via "
        << Pair << "\n";
  return std::move(*Bytes);
}

/// Queue a deferred trampoline for [\p InstOffset, +\p InstSize) with
/// \p Replacement as its body; fixupTrampolineBranches fills in the edges once
/// the pool layout is known. A site beyond s_branch reach of the appended pool
/// uses s_add_pc_i64 on the forward edge. Required rewrites return through an
/// SGPR pair with an SCC-neutral get-PC/add/set-PC sequence; optional rewrites
/// decline. The 8-byte forward branch overwrites the site in place, so a
/// smaller site declines rather than clobbering the next instruction.
[[nodiscard]] bool emitToTrampoline(PatchContext &Ctx, uint64_t InstOffset,
                                    uint32_t InstSize,
                                    ArrayRef<uint8_t> Replacement,
                                    bool AllowSafeFarReturn) {
  // This trampoline lands at the appended pool base and after every trampoline
  // already queued -- later ones are appended behind it and cannot shift it,
  // and fixupTrampolineBranches walks the same list in the same order -- so its
  // final pool offset (relative to .text) is known exactly now.
  uint64_t PoolStart = Ctx.PoolBaseOffset;
  for (const Trampoline &Prev : Ctx.OutTrampolines)
    PoolStart += Prev.Bytes.size();

  // An s_branch encodes To - From as a signed simm16 dword field, in range iff
  // (To - From - MinInstSize) / MinInstSize fits [BranchOffsetMin,
  // BranchOffsetMax] (see LLVMState::encodeSBranch). Test both edges with the
  // short branch-back slot; the branch-back (pool tail -> site) is the farther
  // of the two. Go long only when a short branch cannot reach.
  auto WithinSBranch = [](uint64_t From, uint64_t To) {
    int64_t Dword = (static_cast<int64_t>(To) - static_cast<int64_t>(From) -
                     static_cast<int64_t>(MinInstSize)) /
                    static_cast<int64_t>(MinInstSize);
    return Dword >= BranchOffsetMin && Dword <= BranchOffsetMax;
  };
  const uint64_t ShortBackFrom = PoolStart + Replacement.size();
  const bool Far = !(WithinSBranch(InstOffset, PoolStart) &&
                     WithinSBranch(ShortBackFrom, InstOffset + InstSize));

  Trampoline T;
  T.OriginalOffset = InstOffset;
  T.OriginalSize = InstSize;
  T.Bytes.insert(T.Bytes.end(), Replacement.begin(), Replacement.end());

  if (Far) {
    // HSV-009: the far pool's backward s_add_pc_i64 branch corrupts wave state
    // on gfx1250 A0 (forward is fine). Optional transformations decline the
    // site; required transformations use the SGPR-backed safe return below.
    if (!AllowSafeFarReturn) {
      log() << "hotswap: far trampoline at 0x" << utohexstr(InstOffset)
            << " declined (backward s_add_pc_i64 unreliable on gfx1250 A0, "
            << "HSV-009); site left unpatched\n";
      return false;
    }
    if (InstSize < LongBranchFwdBytes) {
      log() << "hotswap: error: safe far trampoline site 0x"
            << utohexstr(InstOffset) << " is " << InstSize
            << " B, smaller than " << LongBranchFwdBytes
            << " B forward branch\n";
      return false;
    }

    std::optional<uint64_t> BackStart = checkedAddUint64(
        PoolStart, Replacement.size(), "safe far return start offset");
    if (!BackStart)
      return false;
    std::optional<SmallVector<uint8_t>> Back =
        buildSafeFarReturn(Ctx, InstOffset, InstSize, *BackStart);
    if (!Back)
      return false;
    T.Bytes.append(Back->begin(), Back->end());
    T.Long = true;
    T.PreEncodedBack = true;
    Ctx.OutTrampolines.emplace_back(std::move(T));
    return true;
  }
  {
    // Reserve the short branch-back slot; fixupTrampolineBranches fills it in.
    T.Bytes.insert(T.Bytes.end(), MinInstSize, uint8_t{0});
  }
  Ctx.OutTrampolines.emplace_back(std::move(T));
  return true;
}

/// Emit \p Replacement for the instruction at [\p InstOffset,
/// \p InstOffset + \p InstSize). Prefers an in-place NOP-sled rewrite when a
/// reachable sled with sufficient headroom exists; otherwise falls back to a
/// deferred trampoline.
[[nodiscard]] bool emitReplacementCode(PatchContext &Ctx, uint64_t InstOffset,
                                       uint32_t InstSize,
                                       ArrayRef<uint8_t> Replacement,
                                       bool AllowSafeFarReturn) {
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
  return emitToTrampoline(Ctx, InstOffset, InstSize, Replacement,
                          AllowSafeFarReturn);
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
/// records per-kernel stats via ElfView::updateKernelDescriptor.
/// Returns the total number of applied patches across all passes.
static std::optional<uint32_t> applyGfx1250B0toA0Rules(
    std::vector<InternalDecodedInst> &Decoded, uint8_t *Text, uint64_t TextSize,
    const LLVMState &LS, std::vector<Trampoline> &OutTrampolines, ElfView &Elf,
    std::vector<ScratchPatchInfo> &OutScratchPatches,
    const RewriteConfig &Config, bool &OutRequiredPatchApplied) {
  uint32_t Patched = 0;
  std::vector<NopSled> Sleds = buildNopSledMap(Decoded, LS, Elf);

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
  // Pool base as a .text-relative offset for trampoline branch math. The pool
  // is always >= textAddr(); checkedSubUint64 guards a malformed object.
  std::optional<uint64_t> PoolVAddr = Elf.trampolinePoolVAddr();
  if (!PoolVAddr)
    return std::nullopt;
  std::optional<uint64_t> PoolBaseOffset = checkedSubUint64(
      *PoolVAddr, Elf.textAddr(), "trampoline pool base offset");
  if (!PoolBaseOffset)
    return std::nullopt;
  PatchContext Ctx{Config,         Decoded,         Text,
                   TextSize,       *PoolBaseOffset, LS,
                   OutTrampolines, Sleds,           Elf,
                   Liveness,       KernelStats,     OutScratchPatches};

  const HotswapPatchVTable &VT = getHotswapPatchVTable();

  // Skip undecoded slots produced by the decoder for bytes it could not
  // classify as a valid instruction; the dispatcher has nothing to match
  // against on these and we must not invoke the patch passes for them.
  constexpr StringLiteral UnknownMnemonic = "<unknown>";
  using PerInstPatchFn = uint32_t (*)(PatchContext &, size_t);
  SmallVector<PerInstPatchFn, 5> PerInstPasses;
  if (Config.RunB0A0Patches) {
    PerInstPasses.push_back(VT.applyInPlacePatches);
    PerInstPasses.push_back(VT.applyTrampolinePatches);
    PerInstPasses.push_back(VT.applyWmmaSplitPatches);
    PerInstPasses.push_back(VT.applyScratchPatches);
    PerInstPasses.push_back(VT.applyWmmaScale16Patches);
  } else {
    PerInstPasses.push_back(VT.applyTrampolinePatches);
  }

  for (size_t Idx = 0, E = Decoded.size(); Idx < E; ++Idx) {
    const InternalDecodedInst &DI = Decoded[Idx];
    if (DI.Mnemonic == UnknownMnemonic)
      continue;

    for (PerInstPatchFn Fn : PerInstPasses) {
      std::optional<uint32_t> P = runPerInstPass(Fn, Ctx, Idx);
      if (!P)
        return std::nullopt;
      if (*P == 0)
        continue;
      Patched += *P;
      break;
    }
  }

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
  if (Config.RunB0A0Patches && VT.applyWmmaHazardPatch)
    Patched += VT.applyWmmaHazardPatch(Ctx);
  if (Config.RunB0A0Patches && VT.applyVop3px2Src2Fix)
    Patched += VT.applyVop3px2Src2Fix(Ctx);

  for (const llvm::StringMapEntry<KernelPatchStats> &KV : KernelStats) {
    StringRef KName = KV.first();
    const KernelPatchStats &Stats = KV.second;
    if (KName.empty())
      continue;
    std::optional<unsigned> VgprsBefore =
        Elf.getKernelVgprCount(KName, Config.VgprGranuleSize);
    std::optional<unsigned> SgprsBefore = Elf.getKernelSgprCount(KName);
    if (Stats.ExtraVgprs > 0)
      Elf.updateKernelDescriptor(KName, Stats.ExtraVgprs,
                                 Config.VgprGranuleSize);
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
      if (!Elf.updateKernelDescriptorSgprCount(KName, RequiredSgprs,
                                               /*UpdateDescriptor=*/false)) {
        log() << "hotswap: error: failed to update SGPR count for kernel "
              << KName << "\n";
        return std::nullopt;
      }
    }
    std::optional<unsigned> VgprsAfter =
        Elf.getKernelVgprCount(KName, Config.VgprGranuleSize);
    std::optional<unsigned> SgprsAfter = Elf.getKernelSgprCount(KName);
    log() << "hotswap: liveness: kernel " << KName
          << ": vgprs_before=" << VgprsBefore.value_or(0)
          << ", vgprs_after=" << VgprsAfter.value_or(0)
          << ", sgprs_before=" << SgprsBefore.value_or(0)
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
    TrampOffset += T.Bytes.size();

    // Required far patches carry an SCC-neutral s_get_pc_i64/s_add_nc_u64/
    // s_set_pc_i64 return assembled when the trampoline is queued. Other long
    // returns are never queued on gfx1250 A0 because backward s_add_pc_i64 is
    // unsafe.
    if (!T.PreEncodedBack) {
      const uint32_t BackReserve = T.Long ? LongBranchMaxBytes : MinInstSize;
      const uint64_t BackSlot = TP + T.Bytes.size() - BackReserve;
      const uint64_t ReturnTo = T.OriginalOffset + T.OriginalSize;

      SmallVector<uint8_t> BrBack =
          T.Long ? encodeLongBranch(LS, BackSlot, ReturnTo)
                 : LS.encodeSBranch(BackSlot, ReturnTo);
      if (BrBack.empty() || BrBack.size() > BackReserve) {
        log() << "hotswap: error: trampoline branch-back encoding failed at "
                 "0x"
              << utohexstr(T.OriginalOffset) << (T.Long ? " (long)\n" : "\n");
        return false;
      }
      std::memcpy(T.Bytes.data() + T.Bytes.size() - BackReserve, BrBack.data(),
                  BrBack.size());
      for (uint32_t I = BrBack.size(); I + MinInstSize <= BackReserve;
           I += MinInstSize)
        std::memcpy(T.Bytes.data() + T.Bytes.size() - BackReserve + I,
                    LS.SNopBytes.data(), MinInstSize);
    }

    SmallVector<uint8_t> BrFwd =
        T.Long ? encodeLongBranch(LS, T.OriginalOffset, TP)
               : LS.encodeSBranch(T.OriginalOffset, TP);
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

amd_comgr_status_t retargetCodeObject(const void *ElfData, size_t ElfSize,
                                      const TargetIdentifier &TargetIdent,
                                      const Gfx1250RewriteOptions &Options,
                                      std::unique_ptr<MemoryBuffer> &Out) {
  // The dispatcher fetches the patch vtable lazily via
  // getHotswapPatchVTable() inside applyGfx1250B0toA0Rules; the singleton's
  // initializer binds every register*Patch slot on first access, so no
  // explicit install step is needed here.

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

  // Take a working copy so the input is preserved and we have a mutable
  // buffer to parse / patch.
  std::vector<uint8_t> Buf(static_cast<const uint8_t *>(ElfData),
                           static_cast<const uint8_t *>(ElfData) + ElfSize);

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

  LLVMState LS = initLLVM(TargetIdent);
  if (!LS.Valid) {
    log() << "hotswap: error: retargetCodeObject: initLLVM failed "
          << "for CPU '" << TargetIdent.Processor << "'; aborting rewrite.\n";
    return AMD_COMGR_STATUS_ERROR;
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
    if (!decodeTextSection(Text, Elf.textSize(), LS, Decoded)) {
      log() << "hotswap: error: retargetCodeObject: decodeTextSection "
            << "failed on .text (" << Elf.textSize() << " bytes).\n";
      return AMD_COMGR_STATUS_ERROR;
    }

    std::optional<uint32_t> Patched = applyGfx1250B0toA0Rules(
        Decoded, Text, Elf.textSize(), LS, Deferred, Elf, ScratchPatches,
        Config, RequiredPatchApplied);
    if (!Patched)
      return AMD_COMGR_STATUS_ERROR;
    Count = *Patched;
    log() << "hotswap: applied " << Count << " instruction patches\n";
  } else {
    log() << "hotswap: instruction patches disabled for this rewrite\n";
  }

  // gfx1250 revision is recorded per kernel in the AMDGPU metadata note.
  // Running a B0 object on A0 requires retagging that metadata even when no
  // machine instruction needed rewriting. Native A0 code generation preserves
  // s_clause and emits the same instructions as B0 for valid clauses.
  if (Options.RunB0A0Patches && !Elf.updateGfx1250RevisionMetadata("A0"))
    return AMD_COMGR_STATUS_ERROR;

  std::unique_ptr<WritableMemoryBuffer> Result;
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
  if (!Deferred.empty()) {
    if (!fixupTrampolineBranches(Deferred, Text, PoolBaseOffset, LS)) {
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
    std::optional<uint32_t> EntryCount = appendKernelEntryTrampolines(
        Elf, LS, Config.MaxSgprs, Growth, EntryFixups);
    if (!EntryCount)
      return AMD_COMGR_STATUS_ERROR;
    Count += *EntryCount;
  } else {
    log() << "hotswap: kernel-entry trampolines disabled for this rewrite\n";
  }

  if (!Deferred.empty() &&
      !appendDeferredTrampolinePrefetchGuard(Elf, LS, Growth))
    return AMD_COMGR_STATUS_ERROR;

  if (!Growth.empty()) {
    Result = Elf.growWithTrampolines(Growth, LS.SNopBytes);
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
    patchDebugSections(*Result, Deferred, Elf, GrowthTotal);
    if (!rewriteKernelEntryDescriptorOffsets(*Result, PoolVAddr, LS.Cpu,
                                             EntryFixups))
      return AMD_COMGR_STATUS_ERROR;

    // Give each appended entry stub a `<kernel>.stub` symbol so a dispatch
    // whose entry now points at the stub still resolves to a name (e.g. rocgdb
    // `info dispatches`). This grows only the non-alloc .symtab/.strtab and
    // returns a new buffer; failure is non-fatal (the rewritten code object is
    // still correct, just missing the debug-only symbol).
    if (!EntryFixups.empty()) {
      std::unique_ptr<WritableMemoryBuffer> WithSyms =
          addKernelEntryTrampolineSymbols(*Result, Elf.textSectionIndex(),
                                          Elf.textAddr(), Elf.textSize(),
                                          EntryFixups);
      if (WithSyms)
        Result = std::move(WithSyms);
    }
  } else {
    Result = copyOutputBuffer(Buf.data(), ElfSize, "patched");
    if (!Result)
      return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  if (!ScratchPatches.empty())
    runScratchVerification(*Result, LS, ScratchPatches, Config.MaxVgprs);

  Out = std::move(Result);
  return AMD_COMGR_STATUS_SUCCESS;
}

} // namespace hotswap
} // namespace COMGR
