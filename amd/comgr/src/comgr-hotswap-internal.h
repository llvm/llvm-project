//===- comgr-hotswap-internal.h - HotSwap internal types and declarations -===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Internal header for the HotSwap ISA rewriting subsystem. Shared by all
/// comgr-hotswap-*.cpp compilation units. Not part of the public COMGR API.
///
/// Module structure:
///   comgr-hotswap-elf.cpp       ELF parsing, binary helpers, trampoline growth
///   comgr-hotswap-llvm.cpp      LLVM MC infrastructure (disasm/asm/encode)
///   comgr-hotswap-b0a0.cpp      GFX1250 B0-to-A0 policy + public API
///
//===----------------------------------------------------------------------===//

#ifndef COMGR_HOTSWAP_INTERNAL_H
#define COMGR_HOTSWAP_INTERNAL_H

#include "amd_comgr.h"
#include "comgr-env.h"
#include "comgr.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/ELF.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Support/AMDHSAKernelDescriptor.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

namespace COMGR {
namespace hotswap {

// -- Logging ------------------------------------------------------------------
//
// Single output stream for all hotswap diagnostics (errors, warnings, and
// verbose traces). Returns llvm::errs() if AMD_COMGR_EMIT_VERBOSE_LOGS is set
// (via COMGR::env::shouldEmitVerboseLogs()) and llvm::nulls() otherwise, so
// hotswap output stays quiet in normal use but callers can opt in to the full
// diagnostic trail without relinking. Every function that returns a null /
// empty / failure result should emit here with a `"hotswap: error: ..."` or
// `"hotswap: ..."` prefix so the failure path is traceable.

inline llvm::raw_ostream &log() {
  return COMGR::env::shouldEmitVerboseLogs() ? llvm::errs() : llvm::nulls();
}

// -- Trampoline and NOP sled --------------------------------------------------

struct Trampoline {
  uint64_t OriginalOffset = 0;
  uint32_t OriginalSize = 0;
  llvm::SmallVector<uint8_t> Bytes;
};

struct NopSled {
  uint64_t Start = 0;
  uint64_t End = 0;
  uint64_t WritePos = 0;
};

// -- Rewrite rule -------------------------------------------------------------

struct RewriteRule {
  std::string ReplaceMnemonic;
  llvm::SmallVector<uint8_t> ReplaceBytes;
};

// -- Named constants ----------------------------------------------------------

// Kernel descriptor size and RSRC1 offset from upstream
// AMDHSAKernelDescriptor.h.
static constexpr uint64_t KdSize = sizeof(llvm::amdhsa::kernel_descriptor_t);
static constexpr uint64_t KdRsrc1Offset =
    llvm::amdhsa::COMPUTE_PGM_RSRC1_OFFSET;

// Maximum distance (bytes) between an instruction and a NOP sled for the
// sled to be considered reachable by a single s_branch.
static constexpr int64_t MaxSledDistance = 131072;

// Minimum size (bytes) of a consecutive NOP run to be usable as a sled.
static constexpr uint64_t MinNopSledSize = 8;

// Minimum AMDGPU instruction size (one dword).
static constexpr uint32_t MinInstSize = 4;

// s_branch encoding: 16-bit signed dword offset field bounds. Used by
// LLVMState::encodeSBranch to reject out-of-range branches before handing
// them to MCCodeEmitter.
static constexpr int64_t BranchOffsetMin = -32768;
static constexpr int64_t BranchOffsetMax = 32767;

// -- ElfView ------------------------------------------------------------------
//
// Thin wrapper around llvm::object::ELFFile<ELF64LE> that owns the structural
// view of a mutable code-object buffer. The caller retains ownership of the
// bytes; ElfView exposes LLVM's ELF iterators through member methods and
// caches the .text section lookup.

class ElfView {
public:
  using ELFT = llvm::object::ELF64LE;
  using ELFFileT = llvm::object::ELFFile<ELFT>;

  /// Parse \p Data / \p Size into an ElfView. Fails if the bytes are not a
  /// valid ELF64 or if no `.text` section is found.
  static llvm::Expected<ElfView> create(uint8_t *Data, size_t Size);

  ElfView(ElfView &&) = default;
  ElfView &operator=(ElfView &&) = default;
  ElfView(const ElfView &) = delete;
  ElfView &operator=(const ElfView &) = delete;

  const ELFFileT &file() const { return File; }
  size_t size() const { return File.getBufSize(); }

  /// Writable view of the underlying bytes. The caller that constructed this
  /// ElfView via `create(uint8_t *, size_t)` retains ownership of the buffer;
  /// ElfView just exposes a typed, mutable alias onto `ELFFile::base()`. Safe
  /// because the factory was handed a `uint8_t *` and the buffer outlives
  /// this ElfView.
  uint8_t *data() { return const_cast<uint8_t *>(File.base()); }
  const uint8_t *data() const { return File.base(); }

  /// Section header range, cached at construction time. The underlying
  /// storage is the file buffer, which lives at least as long as this
  /// ElfView, so the range is always valid to iterate.
  ELFT::ShdrRange sections() const { return Sections; }

  /// Return the cached `.text` section header. Never null for a successfully
  /// constructed ElfView.
  const ELFT::Shdr *textSection() const { return TextSection; }

  uint64_t textOffset() const { return TextSection->sh_offset; }
  uint64_t textSize() const { return TextSection->sh_size; }
  uint64_t textAddr() const { return TextSection->sh_addr; }

  /// Index of the `.text` section in the section header table.
  unsigned textSectionIndex() const { return TextSectionIndex; }

  /// Pointer into the buffer for the first byte of `.text`.
  uint8_t *textData() { return data() + textOffset(); }
  const uint8_t *textData() const { return data() + textOffset(); }

  /// Find the kernel function symbol whose range includes \p TextOffset.
  /// Returns "" if no matching function symbol exists.
  std::string findKernelAtOffset(uint64_t TextOffset) const;

  /// Pointer to the kernel_descriptor for \p KernelName inside the buffer,
  /// or nullptr if not found.
  uint8_t *findKernelDescriptor(llvm::StringRef KernelName);

  /// Read the VGPR count from the kernel descriptor for \p KernelName.
  /// Returns std::nullopt if the descriptor is not found.
  std::optional<unsigned> getKernelVgprCount(llvm::StringRef KernelName,
                                             unsigned VgprGranuleSize) const;

  /// Update the RSRC1 VGPR/SGPR granule counts in the kernel descriptor for
  /// \p KernelName by adding \p ExtraVgprs / \p ExtraSgprs, using
  /// \p VgprGranuleSize / \p SgprGranuleSize so the call is ISA-agnostic.
  void updateKernelDescriptor(llvm::StringRef KernelName, unsigned ExtraVgprs,
                              unsigned ExtraSgprs, unsigned VgprGranuleSize,
                              unsigned SgprGranuleSize);

  /// Grow the ELF by inserting trampoline bytes after `.text` and adjusting
  /// all section and program headers. Returns a null unique_ptr on failure.
  ///
  /// SHF_ALLOC sections after `.text` (e.g. `.dynamic` in clang/lld-produced
  /// HSACOs) are handled: their file offsets, virtual addresses (sh_addr,
  /// p_vaddr, p_paddr), and segment sizes are shifted by the total
  /// trampoline size to keep the ELF layout consistent.
  std::unique_ptr<llvm::WritableMemoryBuffer>
  growWithTrampolines(llvm::ArrayRef<Trampoline> Trampolines,
                      llvm::ArrayRef<uint8_t> SNopBytes) const;

private:
  ElfView(ELFFileT File, ELFT::ShdrRange Sections,
          const ELFT::Shdr *TextSection, unsigned TextSectionIndex)
      : File(std::move(File)), Sections(Sections), TextSection(TextSection),
        TextSectionIndex(TextSectionIndex) {}

  ELFFileT File;
  ELFT::ShdrRange Sections;
  const ELFT::Shdr *TextSection;
  unsigned TextSectionIndex;
};

// -- Free-function ELF helpers (no ELF state required) ------------------------

/// Overwrite instruction bytes at \p InstOffset with \p Rule.ReplaceBytes,
/// padding remaining bytes with s_nop instructions sourced from \p
/// LS.SNopBytes. Returns false on bounds violation or if \p LS has no cached
/// s_nop encoding.
struct LLVMState;
[[nodiscard]] bool applyByteReplace(const RewriteRule &Rule,
                                    uint64_t InstOffset, uint32_t InstSize,
                                    uint8_t *Text, uint64_t TextSize,
                                    const LLVMState &LS);

/// Find the nearest NOP sled to \p Offset with at least \p Needed bytes of
/// free space. Returns nullptr if none found within MaxSledDistance.
NopSled *findNearestSled(std::vector<NopSled> &Sleds, uint64_t Offset,
                         uint64_t Needed);

// -- RewriteConfig ------------------------------------------------------------
//
// ISA-specific parameters that drive the generic rewriting infrastructure.
// Constructed by the policy layer (e.g. GFX1250 B0-to-A0 in PR #2203) and
// threaded through the MC helpers (buildTrampoline below) and the policy
// PatchContext so infrastructure has zero ISA assumptions.
//
// Instruction-encoding bits (s_branch / s_nop opcodes) are deliberately NOT
// members of this struct -- they are derived from the MC layer at initLLVM()
// time and exposed via LLVMState (SBranchOpcode, SNopBytes plus the
// encodeSBranch method), so the policy layer never has to hardcode target
// opcode values.

struct RewriteConfig {
  std::string SourceIsa;
  std::string TargetIsa;
  std::string TargetCpu;
  unsigned MaxVgprs = 0;
  unsigned VgprGranuleSize = 0;
  unsigned SgprGranuleSize = 0;
};

// -- LLVM MC context ----------------------------------------------------------
//
// Bundle of per-ISA LLVM MC objects. Populated by initLLVM, consumed by the
// decode/encode helpers and by the downstream policy layer. Also caches a
// handful of AMDGPU instruction primitives (s_branch MC opcode, pre-encoded
// s_nop bytes) and exposes the encodeSBranch method -- this keeps all
// target-specific opcode knowledge inside the MC layer and off the policy /
// infrastructure layer.

struct LLVMState {
  const llvm::Target *Target = nullptr;
  std::unique_ptr<llvm::MCRegisterInfo> MRI;
  std::unique_ptr<const llvm::MCAsmInfo> MAI;
  std::unique_ptr<llvm::MCInstrInfo> MCII;
  std::unique_ptr<llvm::MCSubtargetInfo> STI;
  std::unique_ptr<llvm::MCContext> Ctx;
  std::unique_ptr<llvm::MCObjectFileInfo> MOFI;
  std::unique_ptr<llvm::MCDisassembler> MCD;
  std::unique_ptr<llvm::MCInstPrinter> MCIP;
  std::unique_ptr<llvm::MCCodeEmitter> MCE;
  /// Target-provided branch / call / relocation analysis. May be null on
  /// targets that do not implement MCInstrAnalysis; callers must check
  /// before dispatching. Cached here so downstream patch passes can ask
  /// `MIA->isBranch(Inst)` / `isCall(Inst)` / `evaluateBranch(...)` instead
  /// of matching mnemonic strings.
  std::unique_ptr<llvm::MCInstrAnalysis> MIA;
  std::string Cpu;

  /// MC opcode index for `s_branch`, resolved once at initLLVM() via the
  /// asm parser. Used by encodeSBranch() below to construct a fresh MCInst
  /// per call.
  unsigned SBranchOpcode = 0;

  /// MC opcode index for `s_nop`. Resolved via the asm parser at initLLVM()
  /// time so decoded-stream consumers (e.g. buildNopSledMap) can match NOPs
  /// by opcode rather than mnemonic string.
  unsigned SNopOpcode = 0;

  /// Pre-encoded bytes for `s_nop 0` (MinInstSize bytes). Populated at
  /// initLLVM() time via MCCodeEmitter and used by applyByteReplace() and
  /// NOP-sled padding paths instead of a hardcoded encoding.
  llvm::SmallVector<uint8_t, 4> SNopBytes;

  /// Cached `v_nop` MCInst, resolved at initLLVM() time. Used by the WMMA
  /// co-execution hazard patch to build trampolines without string
  /// round-trips.
  llvm::MCInst VNopInst;

  bool Valid = false;

  /// Encode a relative `s_branch` from \p FromOffset to \p ToOffset and
  /// return the MinInstSize encoded bytes. Returns an empty vector if the
  /// delta is unaligned, out of the 16-bit signed dword range, or if this
  /// LLVMState is not valid / has no cached s_branch opcode. Uses
  /// MCCodeEmitter for the encoding so no hardcoded opcode bits appear in
  /// the hotswap code. Empty-on-failure matches the convention used by
  /// encodeMCInst() and assembleSingleInst() so the same idiom applies
  /// uniformly across the MC layer.
  [[nodiscard]] llvm::SmallVector<uint8_t>
  encodeSBranch(uint64_t FromOffset, uint64_t ToOffset) const;
};

// -- Decoded instruction ------------------------------------------------------

struct InternalDecodedInst {
  uint64_t Offset = 0;
  uint32_t Size = 0;
  llvm::MCInst Inst;
  std::string Mnemonic;
};

// -- Function declarations (LLVM MC layer) ------------------------------------

/// Initialize LLVM MC infrastructure for the AMDGPU subtarget described by
/// \p TI (produced by Comgr's parseTargetIdentifier). The triple is built
/// from TI.Arch/Vendor/OS/Environ and features are threaded through to
/// createMCSubtargetInfo so the MC layer sees the same subtarget view the
/// caller asked for. AMDGPU MC registration is delegated to
/// COMGR::ensureLLVMInitialized(); the amdgcn Target lookup itself is cached
/// in a thread-safe function-local static.
LLVMState initLLVM(const TargetIdentifier &TI);

/// Disassemble \p Text into \p Decoded using \p LS. Unknown bytes are encoded
/// as MinInstSize-sized entries with mnemonic "<unknown>".
[[nodiscard]] bool decodeTextSection(const uint8_t *Text, uint64_t TextSize,
                                     const LLVMState &LS,
                                     std::vector<InternalDecodedInst> &Decoded);

/// Assemble a single instruction string, returning its encoded bytes.
llvm::SmallVector<uint8_t> assembleSingleInst(llvm::StringRef AsmStr,
                                              const LLVMState &LS);

/// Assemble \p AsmLines and append a branch-back to the next instruction
/// after the original (\p OriginalOffset + \p OriginalSize). The branch-back
/// is encoded via LLVMState::encodeSBranch, so no ISA-specific opcode needs
/// to flow in from the caller.
Trampoline buildTrampoline(llvm::ArrayRef<std::string> AsmLines,
                           uint64_t OriginalOffset, uint32_t OriginalSize,
                           uint64_t TrampolineTextOffset, const LLVMState &LS);

/// Overload that accepts pre-decoded MCInst instructions directly,
/// encoding them via MCCodeEmitter without a string round-trip.
Trampoline buildTrampoline(llvm::ArrayRef<llvm::MCInst> Insts,
                           uint64_t OriginalOffset, uint32_t OriginalSize,
                           uint64_t TrampolineTextOffset, const LLVMState &LS);

/// Return true iff any register operand of \p WmmaInst overlaps the
/// destination operand of \p ValuInst (for WMMA/VALU co-execution hazard
/// detection). Delegates aliasing to MCRegisterInfo::regsOverlap so
/// sub-registers and tuple aliases are handled without a manual range
/// computation.
bool checkVgprOverlap(const llvm::MCInst &WmmaInst,
                      const llvm::MCInst &ValuInst,
                      const llvm::MCRegisterInfo &MRI);

/// WMMA/SWMMAC A0 vs B0 v_nop spacing requirement.
struct WmmaNopReq {
  int A0Nops = 4;
  int B0Nops = 4;
};

/// Classify the A0/B0 v_nop requirement for a WMMA/SWMMAC mnemonic.
WmmaNopReq classifyWmmaNops(llvm::StringRef Mnemonic);

// -- VGPR liveness types ------------------------------------------------------

/// Per-instruction def/use bitvectors over the VGPR index space. Populated by
/// getInstRegDefUse() during liveness analysis; each bit position corresponds
/// to one VGPR (index matches AMDGPU VGPR numbering, e.g. bit 5 = V5).
struct RegDefUse {
  llvm::BitVector Defs;
  llvm::BitVector Uses;
};

/// A basic block in the decoded-instruction CFG. Offsets are byte offsets
/// into .text; \c InstIndices stores positions in the flat Decoded vector;
/// \c Successors / \c Predecessors are indices into CFG::Blocks.
struct BasicBlock {
  uint64_t StartOffset = 0;
  uint64_t EndOffset = 0;
  llvm::SmallVector<size_t> InstIndices;
  llvm::SmallVector<unsigned> Successors;
  llvm::SmallVector<unsigned> Predecessors;
};

/// Control-flow graph over the decoded instruction stream. \c OffsetToBlock
/// is the inverted index mapping a .text byte offset to its owning block
/// index in \c Blocks, used to resolve branch-target / fall-through edges
/// during CFG construction.
struct CFG {
  std::vector<BasicBlock> Blocks;
  llvm::DenseMap<uint64_t, unsigned> OffsetToBlock;
};

/// Dataflow-liveness result for a kernel's VGPR set. \c LiveBefore[i] and
/// \c LiveAfter[i] are the live-in / live-out bitvectors for Decoded[i].
/// \c Converged is false when the iterative solver hit its iteration cap;
/// callers fall back to a conservative all-VGPRs-live analysis in that case.
struct LivenessInfo {
  std::vector<llvm::BitVector> LiveBefore;
  std::vector<llvm::BitVector> LiveAfter;
  bool Converged = false;
};

/// Allocates scratch VGPRs for a patch point, preferring to reuse dead slots
/// from the kernel's existing allocation before extending the allocation past
/// the kernel descriptor's reported VGPR count. Constructed per patch site
/// with the live-set at that site and the kernel's current / maximum VGPR
/// counts.
struct ScratchAllocator {
  llvm::BitVector LiveAtPoint;
  unsigned KdAllocatedVgprs = 0;
  unsigned NextAboveKd = 0;
  unsigned MaxVgprs = 0;
  unsigned ExtraAllocated = 0;

  ScratchAllocator(const llvm::BitVector &Live, unsigned KdVgprs, unsigned Max)
      : LiveAtPoint(Live), KdAllocatedVgprs(KdVgprs), NextAboveKd(KdVgprs),
        MaxVgprs(Max) {}

  /// Allocate one VGPR not currently marked live. Returns std::nullopt if
  /// the kernel's existing VGPR pool is saturated and there is no headroom
  /// below MaxVgprs for an additional allocation.
  std::optional<unsigned> alloc() {
    for (unsigned V = KdAllocatedVgprs; V-- > 0;) {
      if (!LiveAtPoint.test(V)) {
        LiveAtPoint.set(V);
        return V;
      }
    }
    if (NextAboveKd >= MaxVgprs)
      return std::nullopt;
    unsigned V = NextAboveKd++;
    ExtraAllocated++;
    LiveAtPoint.set(V);
    return V;
  }

  unsigned extraVgprsNeeded() const { return ExtraAllocated; }
};

/// Bookkeeping for a single patch site's scratch allocation. \c Offset is
/// the .text byte offset of the patch; \c ScratchRegs is the bitvector of
/// VGPRs the patch claimed at that site. Consumed by the post-patch
/// verifier (verifyPatchCorrectness) to check the patches are mutually
/// consistent across the kernel.
struct ScratchPatchInfo {
  uint64_t Offset = 0;
  llvm::BitVector ScratchRegs;
};

// -- Patch types --------------------------------------------------------------

/// Per-kernel counters accumulated by the patch passes. Reported via log()
/// at the end of the rewrite and exposed through the public
/// amd_comgr_hotswap_result_t once that result struct is wired up.
struct KernelPatchStats {
  unsigned ExtraVgprs = 0;
  unsigned ScratchReused = 0;
  unsigned ScratchAboveKd = 0;
};

/// Mutable per-run context threaded through all patch passes. Bundles the
/// input config, decoded instruction stream, raw .text bytes, MC state,
/// output streams (trampolines / scratch info), and the shared ELF view +
/// liveness result so patch passes have a single parameter to pass around.
struct PatchContext {
  const RewriteConfig &Config;
  std::vector<InternalDecodedInst> &Decoded;
  uint8_t *Text = nullptr;
  uint64_t TextSize = 0;
  const LLVMState &LS;
  std::vector<Trampoline> &OutTrampolines;
  std::vector<NopSled> &NopSleds;
  ElfView &Elf;
  const LivenessInfo &Liveness;
  llvm::StringMap<KernelPatchStats> &KernelStats;
  std::vector<ScratchPatchInfo> &OutScratchPatches;
};

// -- Function declarations (B0-to-A0 policy layer) ----------------------------

/// Run the full GFX1250 B0-to-A0 rewrite pipeline on \p ElfData / \p ElfSize.
/// \p TargetIdent is the parsed target ISA (produced upstream by Comgr's
/// parseTargetIdentifier()); it is threaded into the MC init so the subtarget
/// triple and feature flags are preserved rather than being reconstructed
/// from just the processor name. On success \p Out is populated with an owned
/// buffer containing the rewritten code object. The caller can transfer the
/// buffer directly to a comgr DataObject via
/// DataObject::setData(std::unique_ptr<MemoryBuffer>).
amd_comgr_status_t
retargetCodeObjectB0A0(const void *ElfData, size_t ElfSize,
                       const TargetIdentifier &TargetIdent,
                       std::unique_ptr<llvm::MemoryBuffer> &Out);

} // namespace hotswap
} // namespace COMGR

#endif // COMGR_HOTSWAP_INTERNAL_H
