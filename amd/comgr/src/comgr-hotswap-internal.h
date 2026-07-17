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
///   comgr-hotswap-occupancy.cpp VGPR/workgroup capacity policy
///   comgr-hotswap-profile.cpp   HotSwap rewrite profiler (out-of-line bodies)
///
//===----------------------------------------------------------------------===//

#ifndef COMGR_HOTSWAP_INTERNAL_H
#define COMGR_HOTSWAP_INTERNAL_H

#include "amd_comgr.h"
#include "comgr-env.h"
#include "comgr.h"
#include "time-stat/ts-interface.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
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
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

namespace COMGR {
namespace hotswap {

class ElfView;

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

inline std::optional<uint64_t> checkedAddUint64(uint64_t LHS, uint64_t RHS,
                                                llvm::StringRef Context) {
  std::optional<uint64_t> Result = llvm::checkedAddUnsigned(LHS, RHS);
  if (Result)
    return Result;

  log() << "hotswap: error: " << Context << " overflows uint64_t.\n";
  return std::nullopt;
}

inline std::optional<uint64_t> checkedSubUint64(uint64_t LHS, uint64_t RHS,
                                                llvm::StringRef Context) {
  if (LHS < RHS) {
    log() << "hotswap: error: " << Context << " underflows uint64_t.\n";
    return std::nullopt;
  }
  return LHS - RHS;
}

// -- HotSwap rewrite profiling -----------------------------------------------
//
// Opt-in via AMD_COMGR_TIME_STATISTICS. When disabled each hook is a single
// branch -- no clock read, no lock -- so no compile-time gate is needed.
//
// retargetCodeObject is single-threaded per call but runs concurrently across
// threads, so each call owns a stack-local HotswapProfile that records into a
// fixed array indexed by HotswapMetric (no lock, no string lookup on the hot
// path) and merges once into Comgr's TimeStatistics when the rewrite finishes.
//
// Row names encode one parent/child level via '/': phase:* pipeline stages
// (timed stages + phase:unaccounted partition phase:rewrite_total), strat:*
// patch strategies with per-rule children, and jump:* placement outcomes.

/// Identity of a profiled bucket. Used as an array index so the hot path
/// records without hashing a string. The enumerator order MUST match the
/// hotswapMetricInfo table below.
enum class HotswapMetric : uint8_t {
  // phase:* rows. The entries flagged PartitionsTotal in hotswapMetricInfo sum,
  // together with Unaccounted, to RewriteTotal. Declared in pipeline order.
  RewriteTotal,
  InputCopy,
  ElfParse,
  InitLLVM,
  Decode,
  B0A0Dispatch,
  PoolSetup,
  FixupTrampolines,
  EntryTrampolines,
  PrefetchGuard,
  GrowElf,
  DebugSections,
  KdRewrite,
  SymbolInsert,
  ScratchVerify,
  OutputCopy,
  Unaccounted,
  // dispatch-internal sub-phases (shown indented under B0A0Dispatch)
  NopSledScan,
  CfgBuild,
  Liveness,
  // strat:* parents
  InPlace,
  Trampoline,
  WmmaSplit,
  ScratchFp8,
  WmmaScale16,
  WmmaHazard,
  Vop3px2Src2,
  // strat:inplace children (s_clause is handled identically on A0/B0 upstream,
  // so it is no longer an in-place rewrite and has no bucket)
  InPlaceClusterLoad,
  InPlaceBarrierSignal,
  // strat:trampoline children
  TrampolineDs2Addr,
  TrampolineTensorTdm,
  TrampolineAddtid,
  TrampolineClusterLoad,
  // jump:* outcomes (count-only rows)
  JumpNopSled,
  JumpShort,
  JumpLong,
  JumpDeclined,
  Count
};

inline constexpr size_t HotswapMetricCount =
    static_cast<size_t>(HotswapMetric::Count);

inline uint64_t profNowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

struct HotswapSample {
  uint64_t Nanos = 0;
  uint64_t Calls = 0;
  uint64_t Patches = 0;
  uint64_t MinNanos = std::numeric_limits<uint64_t>::max();
  uint64_t MaxNanos = 0;
};

/// Static per-metric display info: label, parent (Count == top-level row), and
/// whether the row partitions phase:rewrite_total. Indexed by HotswapMetric;
/// the array order MUST match the enumerator order.
struct HotswapMetricInfo {
  const char *Label;
  HotswapMetric Parent;
  bool PartitionsTotal;
};

inline constexpr HotswapMetricInfo hotswapMetricInfo[HotswapMetricCount] = {
    {"phase:rewrite_total", HotswapMetric::Count, false},
    {"phase:input_copy", HotswapMetric::Count, true},
    {"phase:elf_parse", HotswapMetric::Count, true},
    {"phase:initLLVM", HotswapMetric::Count, true},
    {"phase:decode", HotswapMetric::Count, true},
    {"phase:b0a0_dispatch", HotswapMetric::Count, true},
    {"phase:pool_setup", HotswapMetric::Count, true},
    {"phase:fixup_trampolines", HotswapMetric::Count, true},
    {"phase:entry_trampolines", HotswapMetric::Count, true},
    {"phase:prefetch_guard", HotswapMetric::Count, true},
    {"phase:grow_elf", HotswapMetric::Count, true},
    {"phase:debug_sections", HotswapMetric::Count, true},
    {"phase:kd_rewrite", HotswapMetric::Count, true},
    {"phase:symbol_insert", HotswapMetric::Count, true},
    {"phase:scratch_verify", HotswapMetric::Count, true},
    {"phase:output_copy", HotswapMetric::Count, true},
    {"phase:unaccounted", HotswapMetric::Count, false},
    {"nop_sled_scan", HotswapMetric::B0A0Dispatch, false},
    {"cfg_build", HotswapMetric::B0A0Dispatch, false},
    {"liveness", HotswapMetric::B0A0Dispatch, false},
    {"strat:inplace", HotswapMetric::Count, false},
    {"strat:trampoline", HotswapMetric::Count, false},
    {"strat:wmma_split", HotswapMetric::Count, false},
    {"strat:scratch_fp8", HotswapMetric::Count, false},
    {"strat:wmma_scale16", HotswapMetric::Count, false},
    {"strat:wmma_hazard", HotswapMetric::Count, false},
    {"strat:vop3px2_src2", HotswapMetric::Count, false},
    {"cluster_load_swap", HotswapMetric::InPlace, false},
    {"s_barrier_signal_isfirst", HotswapMetric::InPlace, false},
    {"ds_2addr", HotswapMetric::Trampoline, false},
    {"tensor_tdm", HotswapMetric::Trampoline, false},
    {"addtid", HotswapMetric::Trampoline, false},
    {"cluster_load_mask", HotswapMetric::Trampoline, false},
    {"jump:nop_sled", HotswapMetric::Count, false},
    {"jump:short_s_branch", HotswapMetric::Count, false},
    {"jump:far_set_pc_back", HotswapMetric::Count, false},
    {"jump:declined_far", HotswapMetric::Count, false},
};

/// True when hotswap timings should be recorded (AMD_COMGR_TIME_STATISTICS).
inline bool hotswapProfilingEnabled() { return env::needTimeStatistics(); }

/// Per-rewrite profiling session. Lives on the retargetCodeObject stack and is
/// referenced from PatchContext so deep patch sites record into its lock-free
/// local array. Merges once into Comgr TimeStatistics on destruction.
class HotswapProfile {
public:
  explicit HotswapProfile(bool Enabled) : Enabled(Enabled) {}
  HotswapProfile(const HotswapProfile &) = delete;
  HotswapProfile &operator=(const HotswapProfile &) = delete;
  ~HotswapProfile() {
    if (Enabled)
      flush();
  }

  bool enabled() const { return Enabled; }

  /// RAII timer. Records the elapsed ns (plus any patches) under Metric on
  /// finish() or destruction. A disabled session hands out an inert scope with
  /// a null back-pointer, so the clock is never read on the disabled path.
  class Scope {
  public:
    Scope(HotswapProfile *Profile, HotswapMetric Metric);
    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;
    ~Scope() { finish(); }
    void addPatches(uint64_t P) { Patches += P; }
    void finish();

  private:
    HotswapProfile *Profile;
    HotswapMetric Metric;
    uint64_t StartNs;
    uint64_t Patches = 0;
  };

  [[nodiscard]] Scope time(HotswapMetric Metric);

  /// Count-only record (e.g. jump outcomes): one call, no wall time.
  void count(HotswapMetric Metric, uint64_t N = 1);

  /// Accumulate a pre-measured interval as one call. The pass loop sums locally
  /// and calls this once per rewrite, so a strat:* parent's "calls" stays one.
  void add(HotswapMetric Metric, uint64_t Nanos, uint64_t Patches);

  /// Read-only view of the per-metric samples. Exposed for unit testing the
  /// session accumulation; production code reports through TimeStatistics.
  const HotswapSample &sample(HotswapMetric Metric) const;

  /// Derive phase:unaccounted and convert each recorded sample into a
  /// TimeStatistics::PerfStatRecord (ns -> configured unit, one-level
  /// parent/child row name, e.g. "strat:trampoline/ds_2addr"). Row-name storage
  /// is appended to \p Names, which must outlive the returned records (each
  /// Name is a StringRef into it). flush() merges the result; unit tests
  /// inspect it. Defined in comgr-hotswap-profile.cpp.
  llvm::SmallVector<COMGR::TimeStatistics::PerfStatRecord, HotswapMetricCount>
  buildRecords(llvm::SmallVectorImpl<std::string> &Names);

private:
  /// Merge this rewrite's samples into Comgr TimeStatistics in one batch under
  /// a single lock (see buildRecords). Defined in comgr-hotswap-profile.cpp.
  void flush();

  bool Enabled;
  std::array<HotswapSample, HotswapMetricCount> Samples{};
};

// -- Trampoline and NOP sled --------------------------------------------------

struct Trampoline {
  uint64_t OriginalOffset = 0;
  uint32_t OriginalSize = 0;
  llvm::SmallVector<uint8_t> Bytes;
  // When set, the pool is beyond s_branch reach. The source site branches to a
  // nearby NOP gateway, which uses the scratch-backed gfx12 set-PC sequence
  // to reach the pool without executing s_add_pc_i64.
  bool Long = false;
  bool UsesSetPCBack = false;
  unsigned LongBranchSgprBase = 0;
  bool HasPoolBranchIsland = false;
  uint64_t PoolBranchIslandOffset = 0;
  bool UsesShortBranchForward = false;
  bool UsesDirectSetPCForward = false;
  llvm::SmallVector<uint8_t> DirectSetPCForwardBytes;
  llvm::SmallVector<uint64_t, 4> ForwardBranchIslands;
  uint64_t ForwardBranchTargetOffset = 0;
  bool HasForwardGateway = false;
  uint64_t ForwardGatewayOffset = 0;
  llvm::SmallVector<uint8_t> ForwardGatewayBytes;
  // A far-site run may only be coalesced within one known function. Unknown
  // ranges stay unmerged because adjacent symbols are independent entries.
  bool HasFunctionRange = false;
  uint64_t FunctionStart = 0;
  uint64_t FunctionEnd = 0;
};

struct DisplacementEdit {
  uint64_t Offset = 0;
  uint32_t OriginalSize = 0;
  llvm::SmallVector<uint8_t> ReplacementBytes;
};

enum class DisplacementMapBias {
  BeforeInsertedBytes,
  AfterInsertedBytes,
};

class DisplacementPlan {
public:
  static llvm::Expected<DisplacementPlan>
  create(const ElfView &Elf, llvm::ArrayRef<DisplacementEdit> Edits);

  llvm::ArrayRef<DisplacementEdit> edits() const { return Edits; }
  uint64_t oldTextSize() const { return OldTextSize; }
  uint64_t rawGrowth() const { return RawGrowth; }
  uint64_t paddedGrowth() const { return PaddedGrowth; }
  uint64_t newTextSize() const { return OldTextSize + RawGrowth; }
  uint64_t paddedTextSize() const { return OldTextSize + PaddedGrowth; }
  size_t newElfSize(size_t OldElfSize) const {
    return OldElfSize + PaddedGrowth;
  }

  bool mapOffset(uint64_t OldOffset, DisplacementMapBias Bias,
                 uint64_t &NewOffset) const;
  bool rangeOverlapsReplacement(uint64_t OldOffset, uint64_t Size) const;

  llvm::SmallVector<uint8_t> buildText(llvm::ArrayRef<uint8_t> OldText,
                                       llvm::ArrayRef<uint8_t> SNopBytes) const;

private:
  DisplacementPlan(uint64_t OldTextSize, uint64_t RawGrowth,
                   uint64_t PaddedGrowth, std::vector<DisplacementEdit> Edits)
      : OldTextSize(OldTextSize), RawGrowth(RawGrowth),
        PaddedGrowth(PaddedGrowth), Edits(std::move(Edits)) {}

  uint64_t OldTextSize = 0;
  uint64_t RawGrowth = 0;
  uint64_t PaddedGrowth = 0;
  std::vector<DisplacementEdit> Edits;
};

// Kernel-entry stubs are appended as normal .text growth. Keep each entry on
// the same 256-byte alignment expected by AMDGPU kernel descriptors.
static constexpr uint64_t KernelEntryStubStride = 256;
static constexpr uint64_t KernelEntryInstPrefUnitBytes = 128;
static_assert(KernelEntryStubStride % KernelEntryInstPrefUnitBytes == 0,
              "entry-stub stride must be an integral prefetch span");
static constexpr uint32_t KernelEntryStubInstPrefLines =
    KernelEntryStubStride / KernelEntryInstPrefUnitBytes;

// B0->B0 fast-path stub layout (see comgr-hotswap-entry-trampoline-fast.cpp).
// Pre-encoded gfx1250 stub; the two PC-relative delta immediates and the
// per-kernel scratch SGPR register fields are patched per kernel. All offsets
// are into the 256-byte stub.
static constexpr uint64_t FastEntryStubBodyBytes = 40; // body: to s_set_pc_i64
static constexpr uint64_t FastEntryPrefixBytes = 16;   // global_wb + v_nop
static constexpr uint64_t FastEntryPcBaseOffset = 20;  // s_add (after s_get_pc)
static constexpr uint64_t FastEntryDeltaLoOffset = 24; // s_add_co_u32 imm32
static constexpr uint64_t FastEntryDeltaHiOffset = 32; // s_add_co_ci_u32 imm32

// SGPR register-field byte offsets within the stub body. The scratch pair is
// s[N:N+1] with N = ScratchBase (even). Verified by llvm-mc round-trip on
// gfx1250 (see the encoding table in comgr-hotswap-entry-trampoline-fast.cpp):
//   s_get_pc_i64 sdst         : byte = 0x80 | N
//   s_add_co_u32 src0/sdst    : byte = N
//   s_add_co_ci_u32 src0/sdst : byte = N + 1
//   s_set_pc_i64 src          : byte = N
static constexpr uint64_t FastEntryGetPcSdstOffset = 18;
static constexpr uint64_t FastEntryAddLoSrc0Offset = 20;
static constexpr uint64_t FastEntryAddLoSdstOffset = 22;
static constexpr uint64_t FastEntryAddHiSrc0Offset = 28;
static constexpr uint64_t FastEntryAddHiSdstOffset = 30;
static constexpr uint64_t FastEntrySetPcSrcOffset = 36;

struct KernelDescriptorInfo {
  std::string KernelName;
  uint64_t VAddr = 0;
  int64_t EntryOffset = 0;
};

struct KernelClusterDims {
  unsigned X = 0;
  unsigned Y = 0;
  unsigned Z = 0;
};

struct NopSled {
  uint64_t Start = 0;
  uint64_t End = 0;
  uint64_t WritePos = 0;
  uint64_t FunctionStart = 0;
  uint64_t FunctionEnd = 0;
};

enum class MaskWorkaroundPolicy {
  None,
  A0,
  B0,
};

// -- Rewrite rule -------------------------------------------------------------

struct RewriteRule {
  std::string ReplaceMnemonic;
  llvm::SmallVector<uint8_t> ReplaceBytes;
};

// -- Named constants ----------------------------------------------------------

// Kernel descriptor size from upstream AMDHSAKernelDescriptor.h. Field
// offsets are resolved via offsetof(amdhsa::kernel_descriptor_t, field)
// at the access site so the struct definition stays the single source
// of truth and the *_OFFSET constants do not get spelled out twice.
static constexpr uint64_t KdSize = sizeof(llvm::amdhsa::kernel_descriptor_t);

// Maximum distance (bytes) between an instruction and a NOP sled for the
// sled to be considered reachable by a single s_branch.
static constexpr uint64_t MaxSledDistance = 131072;

// Minimum size (bytes) of a consecutive NOP run to be usable as a sled.
static constexpr uint64_t MinNopSledSize = 8;

// Minimum AMDGPU instruction size (one dword).
static constexpr uint32_t MinInstSize = 4;

// Fixed reservation for the SCC-neutral set-PC sequence.
static constexpr uint32_t SetPcReturnReserveBytes = 20;

static constexpr uint32_t SetPcForwardSequenceBytes = SetPcReturnReserveBytes;
static constexpr uint32_t PoolBranchIslandBytes = MinInstSize;

// s_branch encoding: 16-bit signed dword offset field bounds. Used by
// LLVMState::encodeSBranch to reject out-of-range branches before handing
// them to MCCodeEmitter.
static constexpr int64_t BranchOffsetMin = -32768;
static constexpr int64_t BranchOffsetMax = 32767;

// MCInst operand layout for ds_load_addtid_b32 / ds_store_addtid_b32. Shared
// between the trampoline patch (comgr-hotswap-patch-trampoline.cpp) and the
// unit tests that pin the layout (HotswapMCTest.cpp) so a tablegen change
// upstream is caught in one place.
//   operand 0: vdst (load) / data0 (store) -- VGPR register
//   operand 1: combined offset             -- immediate
//   operand 2: gds                         -- immediate (0 = LDS, 1 = GDS)
static constexpr unsigned AddtidOpReg = 0;
static constexpr unsigned AddtidOpOffset = 1;
static constexpr unsigned AddtidOpGds = 2;

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

  struct FunctionTextRange {
    uint64_t Begin = 0;
    uint64_t End = 0;
    const ELFT::Sym *Symbol = nullptr;
    const ELFT::Shdr *Symtab = nullptr;
  };

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

  /// Enumerate function symbol ranges in `.text` using virtual addresses.
  /// Zero-size symbols extend to the next function symbol or `.text` end.
  std::vector<FunctionTextRange> functionTextRanges() const;

  /// Find the kernel function symbol whose range includes \p TextAddress.
  /// Returns "" if no matching function symbol exists.
  std::string findKernelAtAddress(uint64_t TextAddress) const;

  /// Find the section-relative `.text` range of the function containing
  /// \p TextOffset, or std::nullopt if no sized function symbol covers it.
  std::optional<FunctionTextRange>
  findFunctionTextRangeAtOffset(uint64_t TextOffset) const;

  /// Return a pointer to \p Len bytes at virtual address \p VAddr, resolved
  /// through the allocatable section that contains it (any section, not just
  /// `.text` -- e.g. the appended trampoline pool). Returns nullptr if no
  /// section covers the range or it falls outside the buffer.
  const uint8_t *dataAtVAddr(uint64_t VAddr, uint64_t Len) const;

  /// Pointer to the kernel_descriptor for \p KernelName inside the buffer,
  /// or nullptr if not found.
  uint8_t *findKernelDescriptor(llvm::StringRef KernelName);

  /// Enumerate kernel descriptor symbols named "<kernel>.kd" and read their
  /// current kernel_code_entry_byte_offset values. The returned range remains
  /// valid until this ElfView is destroyed.
  llvm::ArrayRef<KernelDescriptorInfo> kernelDescriptors() const;

  /// Return the virtual address of the kernel descriptor symbol for
  /// \p KernelName, or std::nullopt when the descriptor is not present.
  std::optional<uint64_t>
  getKernelDescriptorVAddr(llvm::StringRef KernelName) const;

  /// Rewrite kernel_code_entry_byte_offset for \p KernelName.
  bool updateKernelDescriptorEntryOffset(llvm::StringRef KernelName,
                                         int64_t NewEntryOffset);

  /// Ensure kernel metadata reserves at least \p RequiredSgprs SGPRs. When
  /// \p UpdateDescriptor is true, also update the pre-gfx10 kernel descriptor
  /// field; it is reserved and must remain unchanged on gfx10+.
  bool updateKernelDescriptorSgprCount(llvm::StringRef KernelName,
                                       unsigned RequiredSgprs,
                                       bool UpdateDescriptor);

  /// Update metadata SGPR counts for every named kernel in one parse and
  /// serialization pass. All requested kernels must be present.
  bool updateKernelMetadataSgprCounts(
      const llvm::StringMap<unsigned> &RequiredSgprs);

  /// Update metadata VGPR counts for every named kernel in one parse and
  /// serialization pass. All requested kernels must be present.
  bool updateKernelMetadataVgprCounts(
      const llvm::StringMap<unsigned> &RequiredVgprs);

  /// Retag every gfx1250 kernel in the AMDGPU metadata note with \p Revision.
  /// The revision strings used by gfx1250 ("A0" and "B0") have equal encoded
  /// size, so this preserves the ELF layout.
  bool updateGfx1250RevisionMetadata(llvm::StringRef Revision);

  /// Read COMPUTE_PGM_RSRC3.INST_PREF_SIZE for \p KernelName.
  std::optional<uint32_t>
  getKernelDescriptorInstPrefSize(llvm::StringRef KernelName,
                                  llvm::StringRef TargetCpu) const;

  /// Rewrite COMPUTE_PGM_RSRC3.INST_PREF_SIZE for \p KernelName.
  bool updateKernelDescriptorInstPrefSize(llvm::StringRef KernelName,
                                          llvm::StringRef TargetCpu,
                                          uint32_t InstPrefLines);

  /// Read the VGPR count from the kernel descriptor for \p KernelName.
  /// Returns std::nullopt if the descriptor is not found.
  std::optional<unsigned> getKernelVgprCount(llvm::StringRef KernelName,
                                             unsigned VgprGranuleSize) const;

  /// Read \c .vgpr_count from the AMDGPU metadata note for \p KernelName.
  std::optional<unsigned>
  getKernelMetadataVgprCount(llvm::StringRef KernelName) const;

  /// Read \c .max_flat_workgroup_size from the AMDGPU metadata note for
  /// \p KernelName. Returns std::nullopt if the note, kernel, or key is absent
  /// or malformed.
  std::optional<unsigned>
  getKernelMaxFlatWorkgroupSize(llvm::StringRef KernelName) const;

  /// Read \c .wavefront_size from the AMDGPU metadata note for \p KernelName.
  /// Returns std::nullopt if the note, kernel, or key is absent or malformed.
  std::optional<unsigned>
  getKernelWavefrontSize(llvm::StringRef KernelName) const;

  /// Read `group_segment_fixed_size` from the kernel descriptor for
  /// \p KernelName, i.e. the **static** (compile-time-fixed) LDS allocation
  /// per work-group in bytes. Returns std::nullopt if the descriptor symbol
  /// is missing.
  ///
  /// This is the only LDS quantity visible in the ELF. Dynamic LDS is
  /// allocated by the host at dispatch time (carried in the AQL packet's
  /// `group_segment_size` and propagated to the device via the
  /// `hidden_dynamic_lds_size` kernarg) and is *not* included here, so the
  /// returned value is a lower bound on the total LDS the kernel may
  /// touch. Callers that need to flag potential overflow of gfx1250 A0's
  /// 16-bit M0 limit can use this as a "definitely exceeds"
  /// check; "static fits, dynamic pushes over" cannot be detected
  /// statically. See AMDGPUUsage "Code Object V3 Kernel Descriptor"
  /// (GROUP_SEGMENT_FIXED_SIZE).
  std::optional<uint32_t>
  getKernelStaticLdsSize(llvm::StringRef KernelName) const;

  /// Read the SGPR count for \p KernelName from the \c amdhsa.kernels
  /// msgpack metadata note (\c .sgpr_count key), falling back to the kernel
  /// descriptor when the metadata note is absent. On GFX10+ the kernel
  /// descriptor's \c GRANULATED_WAVEFRONT_SGPR_COUNT is architecturally
  /// reserved, so metadata is the only reliable source when present.
  /// Returns std::nullopt if the matching metadata is malformed, the kernel is
  /// missing from present metadata, or the descriptor fallback is unavailable.
  std::optional<unsigned> getKernelSgprCount(llvm::StringRef KernelName) const;

  /// Read fixed \c .cluster_dims metadata for \p KernelName when present.
  /// Returns std::nullopt when the metadata note or key is absent, or when the
  /// matching metadata is malformed.
  std::optional<KernelClusterDims>
  getKernelClusterDims(llvm::StringRef KernelName) const;

  /// Ensure the RSRC1 VGPR granule count in the kernel descriptor for
  /// \p KernelName covers \p RequiredVgprs. Returns false if the descriptor
  /// is missing, the granule is invalid, or the required count cannot be
  /// encoded. The SGPR granule field is not updated because it is reserved on
  /// GFX10+.
  bool updateKernelDescriptorVgprCount(llvm::StringRef KernelName,
                                       unsigned RequiredVgprs,
                                       unsigned VgprGranuleSize);

  /// Virtual address at which growWithTrampolines appends the trampoline pool:
  /// the first page-aligned address above every existing allocatable section.
  /// Callers that pre-compute branch/stub targets (B0-to-A0 trampolines,
  /// kernel-entry stubs) must resolve pool positions against this value so the
  /// baked branches land on the pool's final location. Single source of truth
  /// shared with growWithTrampolines. std::nullopt on sh_addr+sh_size overflow.
  std::optional<uint64_t> trampolinePoolVAddr() const;

  /// Grow the ELF by appending the trampoline pool at a fresh virtual address
  /// (trampolinePoolVAddr()) in a new PT_LOAD segment, leaving every existing
  /// section, symbol, and segment in place. Returns a null unique_ptr on
  /// failure.
  ///
  /// Appending (rather than growing `.text` and shifting everything after it)
  /// preserves the absolute/PC-relative addresses baked into a fully-linked
  /// AMDGPU code object, which carries no relocations to fix up.
  std::unique_ptr<llvm::WritableMemoryBuffer>
  growWithTrampolines(llvm::ArrayRef<Trampoline> Trampolines,
                      llvm::ArrayRef<uint8_t> SNopBytes) const;

private:
  enum class KernelSgprCacheState {
    Uninitialized,
    Metadata,
    NoMetadata,
    Error,
  };

  ElfView(ELFFileT File, ELFT::ShdrRange Sections,
          const ELFT::Shdr *TextSection, unsigned TextSectionIndex)
      : File(std::move(File)), Sections(Sections), TextSection(TextSection),
        TextSectionIndex(TextSectionIndex) {}

  llvm::ArrayRef<FunctionTextRange> cachedFunctionTextRanges() const;
  const FunctionTextRange *
  findFunctionTextRangeAtAddress(uint64_t TextAddress) const;
  void initializeKernelDescriptorCache() const;
  void initializeKernelSgprCountCache() const;

  ELFFileT File;
  ELFT::ShdrRange Sections;
  const ELFT::Shdr *TextSection;
  unsigned TextSectionIndex;
  mutable std::optional<std::vector<FunctionTextRange>> FunctionRangeCache;
  mutable std::optional<std::vector<KernelDescriptorInfo>>
      KernelDescriptorCache;
  mutable llvm::StringMap<uint64_t> KernelDescriptorFileOffsetCache;
  mutable llvm::StringMap<uint64_t> KernelDescriptorVAddrCache;
  mutable KernelSgprCacheState SgprCacheState =
      KernelSgprCacheState::Uninitialized;
  mutable llvm::StringMap<std::optional<unsigned>> KernelSgprCountCache;
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
  unsigned MaxSgprs = 0;
  unsigned VgprGranuleSize = 0;
  bool RunB0A0Patches = true;
  MaskWorkaroundPolicy MaskPolicy = MaskWorkaroundPolicy::None;
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

  /// MC opcodes for the kernel-entry stub sequence, resolved once at
  /// initLLVM() time by parsing representative asm snippets. The idempotency
  /// matcher compares decoded opcodes against these cached values instead of
  /// matching disassembled mnemonic strings.
  unsigned GlobalWbOpcode = 0;
  unsigned SGetPcI64Opcode = 0;
  unsigned SAddNcU64Opcode = 0;
  unsigned SAddU32Opcode = 0;
  unsigned SAddcU32Opcode = 0;
  unsigned SSetPcI64Opcode = 0;

  /// MC identities used by far-trampoline relocation analysis. Each opcode is
  /// resolved once through the asm parser so policy code never compares
  /// disassembled mnemonic strings.
  unsigned SClauseOpcode = 0;
  unsigned SDelayAluOpcode = 0;
  unsigned SEndPgmOpcode = 0;
  unsigned SEndPgmSavedOpcode = 0;
  unsigned SAddPcI64Opcode = 0;
  unsigned SCallI64Opcode = 0;
  unsigned SSwapPcI64Opcode = 0;
  unsigned SPrefetchInstPcRelOpcode = 0;
  unsigned SPrefetchDataPcRelOpcode = 0;

  /// SCC, recovered from the implicit definition on a parsed scalar compare.
  /// This avoids scanning target register names in policy code.
  llvm::MCRegister SCCRegister;

  /// VCC super-register, recovered from the destination of a parsed scalar
  /// move. Policy code uses regsOverlap against this cached identity so VCC_LO,
  /// VCC_HI, and tuple aliases are handled by LLVM MC.
  llvm::MCRegister VCCRegister;

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
  bool DecodeSucceeded = false;
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

/// Assemble one non-empty assembly source line, returning its encoded bytes.
/// Target pseudos may expand that source line to more than one MCInst.
llvm::SmallVector<uint8_t> assembleSingleInst(llvm::StringRef AsmStr,
                                              const LLVMState &LS);

/// Assemble a newline-separated instruction sequence, returning its encoded
/// bytes.
llvm::SmallVector<uint8_t> assembleInstructions(llvm::StringRef AsmStr,
                                                const LLVMState &LS);

/// Join \p AsmLines into a single newline-terminated assembly source string,
/// as expected by assembleInstructions.
std::string joinAsmLines(llvm::ArrayRef<std::string> AsmLines);

/// Assemble \p AsmLines and append a branch-back to the next instruction
/// after the original (\p OriginalOffset + \p OriginalSize). The branch-back
/// is encoded via LLVMState::encodeSBranch, so no ISA-specific opcode needs
/// to flow in from the caller.
///
/// NOTE: no production caller remains (WMMA-split now defers edge encoding to
/// emitToTrampoline / fixupTrampolineBranches). Kept only as a self-contained
/// helper exercised by the unit tests; prefer emitToTrampoline for new code.
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

/// Patch the VOP3PX2 scale_src2 field (bits [58:50]) to VGPR0 encoding
/// (0x100) in a 16-byte instruction buffer. Returns true if the field
/// was modified (false if already set to the target value).
bool patchScaleSrc2(uint8_t *InstBytes);

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
struct VgprAllocator {
  llvm::BitVector LiveAtPoint;
  unsigned KdAllocatedVgprs = 0;
  unsigned NextAboveKd = 0;
  unsigned MaxVgprs = 0;
  unsigned ExtraAllocated = 0;

  VgprAllocator(const llvm::BitVector &Live, unsigned KdVgprs, unsigned Max)
      : LiveAtPoint(Live), KdAllocatedVgprs(KdVgprs), NextAboveKd(KdVgprs),
        MaxVgprs(Max) {}

  /// Allocate one VGPR not currently marked live. Returns std::nullopt if
  /// the kernel's existing VGPR pool is saturated and there is no headroom
  /// below MaxVgprs for an additional allocation.
  std::optional<unsigned> alloc() {
    if (int V = LiveAtPoint.find_last_unset_in(0, KdAllocatedVgprs); V != -1) {
      LiveAtPoint.set(V);
      return V;
    }
    if (NextAboveKd >= MaxVgprs)
      return std::nullopt;
    unsigned V = NextAboveKd++;
    ExtraAllocated++;
    LiveAtPoint.set(V);
    return V;
  }

  /// Allocate \p N contiguous VGPRs above the kernel's VGPR count, base rounded
  /// up to \p Align. Scattered dead slots below KD can't guarantee contiguity
  /// or alignment, so this always extends the allocation. Returns the base
  /// VGPR, or std::nullopt if the block would reach MaxVgprs. Since MaxVgprs is
  /// 256 on GFX1250, a block that fits stays in VGPR bank 0, so no
  /// s_set_vgpr_msb switch is needed.
  std::optional<unsigned> allocContiguousAboveKd(unsigned N,
                                                 unsigned Align = 2) {
    unsigned Base = NextAboveKd;
    if (Align > 1 && (Base % Align) != 0)
      Base += Align - (Base % Align);
    if (Base + N > MaxVgprs)
      return std::nullopt;
    ExtraAllocated += (Base + N) - NextAboveKd;
    for (unsigned V = Base; V < Base + N; ++V)
      LiveAtPoint.set(V);
    NextAboveKd = Base + N;
    return Base;
  }

  unsigned extraVgprsNeeded() const { return ExtraAllocated; }
};

/// Allocates scratch SGPRs for a patch point. Unlike VGPRs (which have full
/// dataflow liveness), SGPRs have no liveness analysis, so we always allocate
/// above the kernel descriptor's reported SGPR count. This is conservative
/// but safe: no SGPR currently in use by the kernel can be clobbered.
struct SgprAllocator {
  unsigned KdAllocatedSgprs = 0;
  unsigned NextAboveKd = 0;
  unsigned MaxSgprs = 0;

  SgprAllocator(unsigned KdSgprs, unsigned Max)
      : KdAllocatedSgprs(KdSgprs), NextAboveKd(KdSgprs), MaxSgprs(Max) {}

  /// Allocate one SGPR above the kernel's current count. Returns
  /// std::nullopt if no headroom remains below MaxSgprs.
  std::optional<unsigned> alloc() {
    if (NextAboveKd >= MaxSgprs)
      return std::nullopt;
    return NextAboveKd++;
  }

  unsigned extraSgprsNeeded() const { return NextAboveKd - KdAllocatedSgprs; }
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
  unsigned ExtraSgprs = 0;
  unsigned ScratchReused = 0;
  unsigned ScratchAboveKd = 0;
};

/// Hardware limits needed to determine whether a VGPR allocation can still
/// admit every wave of one maximum-size workgroup.
struct SubtargetOccupancyLimits {
  unsigned EUsPerCU = 0;
  unsigned MaxWavesPerCU = 0;
  unsigned MaxFlatWorkgroupSize = 0;
  unsigned VgprAllocGranule = 0;
  unsigned TotalNumVgprs = 0;
  bool Wave64HalvesVgprCapacity = false;
};

struct WorkgroupCapacity {
  unsigned RequiredWavesPerEU = 0;
  unsigned AchievableWavesPerEU = 0;
};

enum class PatchRequirement {
  Optional,
  Required,
};

enum class VgprBumpDecision {
  Apply,
  Decline,
  Fail,
};

struct KernelWorkgroupMetadata {
  unsigned MaxFlatWorkgroupSize = 0;
  unsigned WavefrontSize = 0;
};

struct SafeSgprUsageSummary {
  bool Valid = true;
  bool UsesVcc = false;
  bool HasCall = false;
  unsigned HighWatermark = 0;
};

struct DirectControlFlowInfo {
  llvm::DenseSet<uint64_t> Targets;
  bool HasUnresolvedTargets = false;
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
  // .text-relative offset at which the appended trampoline pool begins
  // (trampolinePoolVAddr() - textAddr()). Trampoline branch offsets are
  // computed against this, not TextSize, since the pool no longer sits
  // immediately after .text.
  uint64_t PoolBaseOffset = 0;
  const LLVMState &LS;
  std::vector<Trampoline> &OutTrampolines;
  std::vector<NopSled> &NopSleds;
  ElfView &Elf;
  const LivenessInfo &Liveness;
  llvm::StringMap<KernelPatchStats> &KernelStats;
  std::vector<ScratchPatchInfo> &OutScratchPatches;
  const DirectControlFlowInfo &DirectControlFlow;
  // Per-rewrite profiling session (inert unless AMD_COMGR_TIME_STATISTICS is
  // set). Deep patch sites record into its lock-free local array.
  HotswapProfile &Profile;
  // Required patches are transformations whose unpatched original code is
  // unsafe to return when the selected rewrite policy needs the patch.
  bool RequiredPatchFailed = false;
  bool RequiredPatchApplied = false;
  // Sum of the bytes already queued in OutTrampolines. Keeping this in the
  // per-rewrite context makes each new pool-position calculation constant
  // time even for code objects with many thousands of patch sites.
  uint64_t QueuedTrampolineBytes = 0;
  // Safe far-return scratch allocation can be queried at many patch sites in
  // one function. Cache the immutable decoded SGPR usage summaries so each
  // function, and the whole-object fallback, is scanned at most once.
  std::optional<SafeSgprUsageSummary> WholeObjectSgprUsage;
  llvm::DenseMap<std::pair<uint64_t, uint64_t>, SafeSgprUsageSummary>
      FunctionSgprUsage{0};
  // Occupancy checks may run at several patch sites in one kernel. Cache the
  // immutable metadata so the AMDGPU note is parsed at most once per field.
  llvm::StringMap<std::optional<KernelWorkgroupMetadata>>
      WorkgroupMetadataCache;
  llvm::StringMap<unsigned> KernelVgprGranuleCache;
};

/// Return occupancy limits for \p Processor from COMGR's ISA metadata table.
std::optional<SubtargetOccupancyLimits>
getSubtargetOccupancyLimits(llvm::StringRef Processor);

/// Compute the VGPR-limited capacity after allocation rounding and the waves
/// per EU needed to admit one maximum-size workgroup. Returns std::nullopt for
/// invalid or unsupported inputs.
std::optional<WorkgroupCapacity>
computeWorkgroupCapacity(unsigned Vgprs, unsigned MaxFlatWorkgroupSize,
                         unsigned WavefrontSize,
                         const SubtargetOccupancyLimits &Limits);

/// Apply when capacity is preserved; otherwise decline optional patches and
/// fail required patches.
VgprBumpDecision decideVgprBump(PatchRequirement Requirement,
                                const WorkgroupCapacity &Capacity);

/// Return the descriptor/allocation granule for the kernel's wavefront mode.
/// Falls back to RewriteConfig::VgprGranuleSize for metadata-free objects; a
/// growth check will still decline those objects because capacity is unknown.
unsigned getKernelVgprGranuleSize(PatchContext &Ctx,
                                  llvm::StringRef KernelName);

/// Preflight a patch site's aggregate VGPR demand before it emits bytes.
VgprBumpDecision checkKernelVgprBump(PatchContext &Ctx,
                                     llvm::StringRef KernelName,
                                     unsigned ExtraVgprs,
                                     PatchRequirement Requirement);

/// A block of numbered SGPRs that is not referenced in the function being
/// patched, or anywhere in the code object when the site may be reached by a
/// call whose register requirements cannot be bounded locally.
struct SafeSgprScratchBlock {
  unsigned Base = 0;
  unsigned Count = 0;
};

/// Find an aligned block of unused numbered SGPRs for \p TextOffset. Returns
/// nullopt after logging when no block fits below RewriteConfig::MaxSgprs.
std::optional<SafeSgprScratchBlock>
findSafeSgprScratchBlock(PatchContext &Ctx, uint64_t TextOffset, unsigned Count,
                         unsigned Alignment, llvm::StringRef Context);

/// Charge a previously selected global block to the kernel owning \p
/// TextOffset. If the site is in an ordinary device function, conservatively
/// charge every kernel descriptor because the ELF does not carry a complete
/// call graph.
bool commitSafeSgprScratchBlock(PatchContext &Ctx, uint64_t TextOffset,
                                const SafeSgprScratchBlock &Block,
                                llvm::StringRef Context);

// -- Trampoline emission helpers (defined in comgr-hotswap-b0a0.cpp) ----------

[[nodiscard]] bool emitToNopSled(PatchContext &Ctx, NopSled &Sled,
                                 uint64_t InstOffset, uint32_t InstSize,
                                 llvm::ArrayRef<uint8_t> Replacement);
[[nodiscard]] bool emitToTrampoline(PatchContext &Ctx, uint64_t InstOffset,
                                    uint32_t InstSize,
                                    llvm::ArrayRef<uint8_t> Replacement);

/// Encode an SCC-neutral indirect long branch using the aligned SGPR pair at
/// \p SgprBase. The displacement uses gfx12's s_add_nc_u64; no s_add_pc_i64
/// is emitted.
std::optional<llvm::SmallVector<uint8_t>>
encodeSetPCLongBranch(const LLVMState &LS, uint64_t FromOffset,
                      uint64_t TargetOffset, unsigned SgprBase);

struct EncodedSetPcGateway {
  NopSled *Sled = nullptr;
  llvm::SmallVector<uint8_t> Bytes;
};

/// Find the nearest short-branch-reachable gateway whose remaining space fits
/// the set-PC sequence encoded for that candidate's address. The returned
/// plan does not advance the sled or modify text.
llvm::Expected<std::optional<EncodedSetPcGateway>>
findNearestSetPcGateway(std::vector<NopSled> &Gateways, const LLVMState &LS,
                        uint64_t FromOffset, uint64_t TargetOffset,
                        unsigned SgprBase);

/// Count set-PC gateway slots reachable from \p FromOffset, up to \p MaxSlots.
/// Zero means that no candidate fits; an Error means that a reachable
/// candidate could not be encoded.
llvm::Expected<uint64_t> countReachableSetPcGatewaySlots(
    llvm::ArrayRef<NopSled> Gateways, const LLVMState &LS, uint64_t FromOffset,
    uint64_t TargetOffset, unsigned SgprBase, uint64_t MaxSlots);

/// Return whether an s_branch at \p From can encode \p To, including the
/// instruction-relative PC base, alignment, signed range, and overflow checks.
bool isSBranchReachable(uint64_t From, uint64_t To);

/// Evaluate a statically direct branch or call target from its decoded MCInst.
/// Uses MCInstrAnalysis where supported and the documented gfx1250 s_call_i64
/// operand fallback otherwise.
std::optional<uint64_t>
evaluateDirectControlFlowTarget(const InternalDecodedInst &DI,
                                const LLVMState &LS);

/// Collect branch and call targets used to protect interior entry points from
/// trampoline coalescing. Absolute addresses in TextAddr .. TextAddr +
/// TextSize are converted to text-relative offsets. Canonical PC-materialized
/// register calls are resolved when their target value has one provable
/// straight-line definition and no direct, indirect, or declared entry can
/// bypass that definition. Canonical local-function returns are bounded to
/// the continuations of calls that preserve the same link register, provided
/// no interior call, overlapping external alias, or reachable fallthrough can
/// enter the function without that link definition. \p DeclaredEntries
/// contains text-relative function and kernel entry offsets; \p FunctionRanges
/// supplies the symbol ranges used for the return proof; \p ExternalEntries
/// identifies externally reachable symbol and kernel entries, including
/// aliases at a local function's start. Other register-target control flow
/// sets HasUnresolvedTargets so callers can disable transformations that
/// consume possible destinations.
std::optional<DirectControlFlowInfo> collectDirectBranchTargets(
    llvm::ArrayRef<InternalDecodedInst> Decoded, const LLVMState &LS,
    uint64_t TextAddr, uint64_t TextSize,
    llvm::ArrayRef<uint64_t> DeclaredEntries,
    llvm::ArrayRef<ElfView::FunctionTextRange> FunctionRanges = {},
    llvm::ArrayRef<uint64_t> ExternalEntries = {});

[[nodiscard]] bool emitReplacementCode(PatchContext &Ctx, uint64_t InstOffset,
                                       uint32_t InstSize,
                                       llvm::ArrayRef<uint8_t> Replacement);

/// Expand one decoded gfx1250 DS two-address instruction into the ordered
/// single-address instruction sequence used by the trampoline patch. Exposed
/// from the internal interface so unit tests can verify read-before-write
/// dependency handling without constructing a complete ELF rewrite. Returns
/// std::nullopt after logging a malformed or unsafe expansion.
[[nodiscard]] std::optional<std::vector<std::string>>
expandDs2Addr(const llvm::MCInst &Inst, llvm::StringRef FromMnem,
              llvm::StringRef ToMnem, const LLVMState &LS);

// -- Patch dispatch vtable ----------------------------------------------------
//
// Function-pointer dispatch table that replaces the prior LLVM_ATTRIBUTE_WEAK
// + `#if !defined(_MSC_VER)` override pattern. PE/COFF does not honour weak
// the way ELF does, so on Windows the weak stubs silently won every patch
// call and the feature was a no-op (issue ROCm/llvm-project#2479).
//
// Patch modules supply their implementations through register*Patch
// functions invoked by installHotswapPatches(). The membership list is
// comgr-hotswap-patches.def; each entry there corresponds to one slot
// below and one register*Patch function in a sibling
// comgr-hotswap-patch-*.cpp. nullptr slots are treated as no-op by the
// dispatcher, so an unmigrated pass family (e.g. scratch) is safe to
// leave unbound until its first strong override lands.
//
// The singleton accessor below eagerly installs every registered slot in
// its own initializer, so production callers never observe an empty
// vtable. installHotswapPatches() is still exported for unit tests that
// want to drive the install against a local HotswapPatchVTable.

struct HotswapPatchVTable {
  // Per-instruction passes: called in declaration order; first non-zero
  // return wins for an instruction (matches the pre-vtable dispatcher
  // behaviour in applyGfx1250B0toA0Rules).
  uint32_t (*applyInPlacePatches)(PatchContext &, size_t) = nullptr;
  uint32_t (*applyTrampolinePatches)(PatchContext &, size_t) = nullptr;
  uint32_t (*applyWmmaSplitPatches)(PatchContext &, size_t) = nullptr;
  uint32_t (*applyScratchPatches)(PatchContext &, size_t) = nullptr;
  uint32_t (*applyWmmaScale16Patches)(PatchContext &, size_t) = nullptr;

  // Whole-kernel passes: called once per kernel after the per-instruction
  // loop completes.
  uint32_t (*applyWmmaHazardPatch)(PatchContext &) = nullptr;
  uint32_t (*applyVop3px2Src2Fix)(PatchContext &) = nullptr;
};

/// Walk comgr-hotswap-patches.def and bind every patch module's
/// implementation into \p VT by calling its register*Patch function.
/// A missing register*Patch produces a link error, which is the
/// loud-failure shape the weak-symbol pattern lacked. Production code
/// never calls this directly; it runs inside getHotswapPatchVTable()'s
/// initializer. Exposed here so unit tests can drive the install against
/// a local HotswapPatchVTable.
void installHotswapPatches(HotswapPatchVTable &VT);

/// Process-wide HotswapPatchVTable singleton (Meyers-style). The
/// initializer eagerly calls installHotswapPatches() on its own storage,
/// so every reference returned here is to a fully bound vtable. C++11
/// [stmt.dcl]/4 guarantees the initializer runs exactly once and is safe
/// under concurrent first access, which removes the need for an explicit
/// std::call_once at the entry point and any inter-TU static-init order
/// contract on the patch modules.
HotswapPatchVTable &getHotswapPatchVTable();

// Forward-declare every patch module's installer from the central .def
// registry. Patch modules define these in their comgr-hotswap-patch-*.cpp;
// installHotswapPatches() consumes them; unit tests under test-unit/ also
// invoke them directly. A patches.def line with no matching definition
// produces a libamd_comgr / HotswapMCTests link error.
#define HOTSWAP_PATCH(Name) void register##Name##Patch(HotswapPatchVTable &);
#include "comgr-hotswap-patches.def"
#undef HOTSWAP_PATCH

// -- Function declarations (kernel-entry trampoline pass) ---------------------

struct KernelEntryTrampolineFixup {
  std::string KernelName;
  uint64_t StubTextOffset = 0;
  unsigned RequiredSgprs = 0;
  uint32_t InstPrefLines = 0;
  // Both the MC path and the fast path allocate a per-kernel scratch pair above
  // the kernel's live SGPR count and bump the descriptor SGPR reservation, so
  // this is normally false. It stays reserved for a caller that installs a stub
  // using a pair already counted in the reservation (no bump needed).
  bool SkipSgprReservation = false;
};

/// Build a 256-byte, entry-aligned HotSwap kernel-entry stub at
/// \p StubVAddr that jumps to \p EntryVAddr using PC-relative address
/// materialization. Returns an empty vector if MC assembly fails.
llvm::SmallVector<uint8_t> buildKernelEntryTrampoline(uint64_t StubVAddr,
                                                      uint64_t EntryVAddr,
                                                      unsigned ScratchSgpr,
                                                      const LLVMState &LS);

/// Structural matcher for the entry stubs produced by
/// buildKernelEntryTrampoline, used to keep the rewrite idempotent.
bool isKernelEntryTrampoline(llvm::ArrayRef<uint8_t> Bytes,
                             const LLVMState &LS);

/// Cheap raw-byte prefilter for the entry stubs produced by
/// buildKernelEntryTrampoline. This is intentionally weaker than
/// isKernelEntryTrampoline and exists to avoid running the disassembler over
/// arbitrary original kernel entry bytes during idempotency checks.
bool hasKernelEntryTrampolinePrefix(llvm::ArrayRef<uint8_t> Bytes,
                                    const LLVMState &LS);

/// Compute the trailing readable guard needed after an appended kernel-entry
/// stub pool so CP instruction prefetches from the last stub cannot run past
/// mapped .text bytes.
uint64_t computeKernelEntryPrefetchGuardBytes(uint32_t InstPrefLines);

/// Queue one direct insertion of `global_wb; v_nop` at each kernel descriptor
/// entry that does not already target either a direct entry prefix or an
/// appended HotSwap entry stub.
std::optional<uint32_t>
collectKernelEntryDisplacements(const ElfView &Elf, const LLVMState &LS,
                                std::vector<DisplacementEdit> &OutEdits);

/// Append one entry stub per kernel descriptor that does not already target a
/// HotSwap entry stub. The stubs are appended to \p Growth and descriptor
/// rewrites are recorded in \p OutFixups for application after ELF growth.
std::optional<uint32_t> appendKernelEntryTrampolines(
    const ElfView &Elf, const LLVMState &LS, unsigned MaxSgprs,
    std::vector<Trampoline> &Growth,
    std::vector<KernelEntryTrampolineFixup> &OutFixups);

/// Apply descriptor rewrites recorded by appendKernelEntryTrampolines after
/// the ELF has been grown.
bool rewriteKernelEntryDescriptorOffsets(
    llvm::WritableMemoryBuffer &OutBuf, uint64_t PoolVAddr,
    llvm::StringRef TargetCpu,
    llvm::ArrayRef<KernelEntryTrampolineFixup> Fixups);

/// Resolve the virtual address of a kernel descriptor's entry point. Shared by
/// the MC and fast entry-trampoline paths.
std::optional<uint64_t> entryVAddr(const KernelDescriptorInfo &KD);

/// Compute LHS - RHS when the result is representable as int64_t.
std::optional<int64_t> checkedSignedDifference(uint64_t LHS, uint64_t RHS,
                                               llvm::StringRef Context);

/// Round Value up to a multiple of Alignment, reporting overflow against
/// Context. Shared by the MC and fast entry-trampoline paths.
std::optional<uint64_t> checkedAlignTo(uint64_t Value, uint64_t Alignment,
                                       llvm::StringRef Context);

/// B0->B0 FAST PATH (comgr-hotswap-entry-trampoline-fast.cpp): emit entry stubs
/// from a pre-encoded gfx1250 byte template with no LLVM MC layer. Same
/// append/fixup contract as appendKernelEntryTrampolines: the scratch pair is
/// allocated per kernel above its live SGPR count and \p ScratchSgpr is patched
/// into the stub's SGPR register fields, so the descriptor SGPR reservation is
/// bumped exactly like the MC path. Selected automatically for pure B0->B0
/// entry-only rewrites.
llvm::SmallVector<uint8_t> buildKernelEntryTrampolineFast(uint64_t StubVAddr,
                                                          uint64_t EntryVAddr,
                                                          unsigned ScratchSgpr);
std::optional<uint32_t> appendKernelEntryTrampolinesFast(
    const ElfView &Elf, llvm::StringRef TargetCpu, unsigned MaxSgprs,
    std::vector<Trampoline> &Growth,
    std::vector<KernelEntryTrampolineFixup> &OutFixups);

/// Add a `<kernel_name>.stub` STT_FUNC symbol to the code object's `.symtab`
/// for each appended kernel-entry stub, so tools that resolve a dispatch's
/// entry address to a name (e.g. rocgdb `info dispatches`, which reads the
/// non-alloc `.symtab`) report the stub instead of a bare address. Returns a
/// newly allocated buffer with the grown `.symtab` / `.strtab`, or nullptr if
/// no symbols were added (empty fixups, missing `.symtab`, or a structural
/// problem) -- callers treat nullptr as "keep the existing buffer", since the
/// symbol is a debugging aid and its absence is not a correctness failure.
///
/// Only the trailing non-alloc `.symtab` / `.strtab` sections grow, so no
/// virtual addresses, program headers, or relocations change; `.dynsym` (used
/// by the loader) is left untouched.
/// \p PoolVAddr is the virtual address of the appended trampoline pool (the
/// same base rewriteKernelEntryDescriptorOffsets() redirects each descriptor
/// to); each stub symbol is placed at PoolVAddr + StubTextOffset in the pool's
/// section, so it matches the address the dispatch entry now targets.
std::unique_ptr<llvm::WritableMemoryBuffer> addKernelEntryTrampolineSymbols(
    llvm::WritableMemoryBuffer &In, uint64_t PoolVAddr,
    llvm::ArrayRef<KernelEntryTrampolineFixup> Fixups);

/// Apply direct .text displacement to a newly allocated output buffer.
llvm::Expected<std::unique_ptr<llvm::WritableMemoryBuffer>>
tryApplyTextDisplacementToNewBuffer(const ElfView &Elf, const LLVMState &LS,
                                    llvm::ArrayRef<DisplacementEdit> Edits);

// -- Function declarations (GFX1250 hotswap policy layer) ---------------------

struct Gfx1250RewriteOptions {
  bool RunB0A0Patches = true;
  bool RunEntryTrampolines = false;
  MaskWorkaroundPolicy MaskPolicy = MaskWorkaroundPolicy::None;
  // Source and target are both gfx1250 B0. Only then may the entry-trampoline
  // rewrite take the no-MC fast path; the source/target stepping needed to
  // decide this is known to the caller but not recoverable from TargetIdent
  // alone. A0->A0 and A0->B0 also run without instruction patches, so this must
  // be set explicitly rather than inferred inside retargetCodeObject.
  bool UseB0B0EntryFastPath = false;
};

/// Run the selected GFX1250 hotswap rewrite passes on \p ElfData / \p ElfSize.
/// \p TargetIdent is the parsed target ISA (produced upstream by Comgr's
/// parseTargetIdentifier() or the hotswap-local stepping parser); it is
/// threaded into the MC init so the subtarget triple and feature flags are
/// preserved rather than being reconstructed from just the processor name. On
/// success \p Out is populated with an owned buffer containing the rewritten
/// code object. The caller can transfer the buffer directly to a comgr
/// DataObject via DataObject::setData(std::unique_ptr<MemoryBuffer>).
amd_comgr_status_t retargetCodeObject(const void *ElfData, size_t ElfSize,
                                      const TargetIdentifier &TargetIdent,
                                      const Gfx1250RewriteOptions &Options,
                                      std::unique_ptr<llvm::MemoryBuffer> &Out);

} // namespace hotswap
} // namespace COMGR

#endif // COMGR_HOTSWAP_INTERNAL_H
