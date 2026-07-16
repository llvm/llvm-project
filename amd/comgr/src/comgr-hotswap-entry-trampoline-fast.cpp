//===- comgr-hotswap-entry-trampoline-fast.cpp - B0->B0 fast path ---------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// B0-on-B0 kernel-entry trampoline FAST PATH. No LLVM MC layer (no initLLVM,
/// no assembler, no disassembler): the stub is emitted from a pre-encoded
/// gfx1250 byte template with the two PC-relative delta immediates and the
/// per-kernel scratch SGPR register fields patched in. Like the MC path, the
/// scratch pair is allocated above each kernel's SGPR count (never a live
/// kernel input, including preloaded kernargs) and the descriptor's SGPR
/// reservation is bumped accordingly. Idempotency and the
/// compile-time-workaround skip are decided by raw byte comparison rather than
/// decoding.
///
/// This path is selected automatically for pure B0->B0 entry-only rewrites
/// (no B0->A0 instruction patches, no mask workaround). The MC-based path in
/// comgr-hotswap-entry-trampoline.cpp handles A0 and any rewrite that needs
/// instruction patches.
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Endian.h"

#include <algorithm>
#include <cstring>
#include <limits>

using namespace llvm;

namespace COMGR {
namespace hotswap {

// Pre-encoded gfx1250 stub template. Rather than run the MC layer at rewrite
// time (the point of the fast path), the stub is emitted from these bytes. To
// keep them from silently drifting from what the assembler produces -- and to
// satisfy the "no hand-maintained encoded byte sequences" convention -- the
// arrays live in a GENERATED .inc built by utils/gen-hotswap-fast-stub-inc.sh
// from utils/comgr-hotswap-entry-trampoline-fast-stub.s, and
// StubTemplateMatchesMCOutput (HotswapMCTest) proves they still equal fresh MC
// output. The 40-byte body is padded to KernelEntryStubStride (256) with
// s_code_end. s_get_pc_i64 loads the address of the instruction after it
// (s_add, at StubVAddr + FastEntryPcBaseOffset), the base for the PC-relative
// delta. The body is spelled with s[100:101]; buildKernelEntryTrampolineFast
// rewrites the six SGPR register-field bytes per kernel to the allocated
// scratch pair (see the FastEntry*Offset encoding table in
// comgr-hotswap-internal.h).
#include "comgr-hotswap-entry-trampoline-fast-stub.inc"

static_assert(sizeof(StubTemplate) == FastEntryStubBodyBytes,
              "generated stub template must be the 40-byte body");

// global_wb; v_nop prefix (first FastEntryPrefixBytes of the body), for raw
// idempotency / workaround detection. Aliased from StubTemplate so the prefix
// can never diverge from the template it is a prefix of.
static constexpr const uint8_t *EntryPrefix = StubTemplate;
static_assert(sizeof(StubTemplate) >= FastEntryPrefixBytes,
              "stub template must contain the workaround prefix");

SmallVector<uint8_t> buildKernelEntryTrampolineFast(uint64_t StubVAddr,
                                                    uint64_t EntryVAddr,
                                                    unsigned ScratchSgpr) {
  SmallVector<uint8_t> Bytes;
  Bytes.resize(KernelEntryStubStride);
  std::memcpy(Bytes.data(), StubTemplate, FastEntryStubBodyBytes);

  // Patch the scratch SGPR pair s[N:N+1] into the register fields. The template
  // is spelled with s[100:101]; only the six field bytes change with N (see the
  // FastEntry*Offset encoding table in comgr-hotswap-internal.h). ScratchSgpr
  // is an even base <= 104 (guaranteed by the aligned-pair allocation in
  // appendKernelEntryTrampolinesFast).
  const uint8_t N = static_cast<uint8_t>(ScratchSgpr);
  Bytes[FastEntryGetPcSdstOffset] = 0x80 | N;
  Bytes[FastEntryAddLoSrc0Offset] = N;
  Bytes[FastEntryAddLoSdstOffset] = N;
  Bytes[FastEntryAddHiSrc0Offset] = N + 1;
  Bytes[FastEntryAddHiSdstOffset] = N + 1;
  Bytes[FastEntrySetPcSrcOffset] = N;

  // Materialize EntryVAddr relative to the s_get_pc base. Two's complement, so
  // back-jumps are handled.
  const uint64_t PcBase = StubVAddr + FastEntryPcBaseOffset;
  const uint64_t Delta = EntryVAddr - PcBase;
  llvm::support::endian::write32le(Bytes.data() + FastEntryDeltaLoOffset,
                                   static_cast<uint32_t>(Delta));
  llvm::support::endian::write32le(Bytes.data() + FastEntryDeltaHiOffset,
                                   static_cast<uint32_t>(Delta >> 32));

  // Pad to stride with s_code_end (prefetch-safe; never executed).
  for (uint64_t Off = FastEntryStubBodyBytes; Off < KernelEntryStubStride;
       Off += sizeof(SCodeEnd))
    std::memcpy(Bytes.data() + Off, SCodeEnd, sizeof(SCodeEnd));
  return Bytes;
}

// Raw byte check: does the descriptor's current entry already begin with
// global_wb; v_nop (either a hotswap fast stub already installed, or the
// compile-time unclaused-VMEM workaround prologue)? Both mean "do not add a
// trampoline".
static std::optional<bool>
entryHasWorkaroundPrefixFast(const ElfView &Elf,
                             const KernelDescriptorInfo &KD) {
  std::optional<uint64_t> Entry = entryVAddr(KD);
  if (!Entry)
    return std::nullopt;
  const uint8_t *EntryBytes = Elf.dataAtVAddr(*Entry, FastEntryPrefixBytes);
  if (!EntryBytes) {
    log()
        << "hotswap: fast: kernel '" << KD.KernelName << "' entry vaddr 0x"
        << Twine::utohexstr(*Entry)
        << " is not backed by readable data; assuming no workaround prefix.\n";
    return false;
  }
  return std::memcmp(EntryBytes, EntryPrefix, FastEntryPrefixBytes) == 0;
}

// Allocate an aligned scratch SGPR pair s[N:N+1] just above the kernel's live
// SGPR count, exactly like the MC path's allocateEntryStubScratchSgprs. This is
// the correctness guarantee over a fixed s[100:101]: N is above every live
// input (system/user SGPRs and preloaded kernargs), so the stub never clobbers
// one. Declines (returns nullopt) if no aligned pair fits below MaxSgprs.
static std::optional<unsigned> allocateEntryStubScratchSgprsFast(
    const ElfView &Elf, const KernelDescriptorInfo &KD, unsigned MaxSgprs) {
  constexpr unsigned ScratchSgprs = 2;
  std::optional<unsigned> SgprCount = Elf.getKernelSgprCount(KD.KernelName);
  if (!SgprCount) {
    log() << "hotswap: error: fast entry trampoline: failed to read SGPR count "
          << "for '" << KD.KernelName << "'.\n";
    return std::nullopt;
  }
  if (*SgprCount > MaxSgprs) {
    log() << "hotswap: error: fast entry trampoline: kernel '" << KD.KernelName
          << "' uses " << *SgprCount << " SGPRs, above max " << MaxSgprs
          << ".\n";
    return std::nullopt;
  }

  unsigned ScratchBase = (*SgprCount + 1) & ~1u;
  if (ScratchBase > MaxSgprs || MaxSgprs - ScratchBase < ScratchSgprs) {
    log() << "hotswap: error: fast entry trampoline: kernel '" << KD.KernelName
          << "' uses " << *SgprCount << " SGPRs; no aligned scratch pair fits "
          << "below max " << MaxSgprs << ".\n";
    return std::nullopt;
  }
  return ScratchBase;
}

static bool appendPaddingFast(std::vector<Trampoline> &Out, uint64_t PadBytes) {
  if (PadBytes == 0)
    return true;
  if (PadBytes % sizeof(SNop) != 0) {
    log() << "hotswap: error: fast entry-stub padding " << PadBytes
          << " is not a multiple of s_nop size.\n";
    return false;
  }
  Trampoline Pad;
  Pad.Bytes.reserve(PadBytes);
  while (static_cast<uint64_t>(Pad.Bytes.size()) < PadBytes)
    Pad.Bytes.append(SNop, SNop + sizeof(SNop));
  Out.push_back(std::move(Pad));
  return true;
}

std::optional<uint32_t> appendKernelEntryTrampolinesFast(
    const ElfView &Elf, StringRef TargetCpu, unsigned MaxSgprs,
    std::vector<Trampoline> &Growth,
    std::vector<KernelEntryTrampolineFixup> &OutFixups) {
  ArrayRef<KernelDescriptorInfo> Descriptors = Elf.kernelDescriptors();
  if (Descriptors.empty())
    return 0;

  struct WorkItem {
    KernelDescriptorInfo KD;
    uint32_t StubInstPrefLines = 0;
  };
  std::vector<WorkItem> Work;
  uint32_t MaxStubInstPrefLines = 0;

  for (const KernelDescriptorInfo &KD : Descriptors) {
    // Skip if the entry already carries the workaround (already-installed fast
    // stub, or a compile-time global_wb; v_nop prologue). Raw byte check.
    std::optional<bool> HasPrefix = entryHasWorkaroundPrefixFast(Elf, KD);
    if (!HasPrefix)
      return std::nullopt;
    if (*HasPrefix) {
      log() << "hotswap: fast: kernel '" << KD.KernelName
            << "' entry already has global_wb; v_nop; skipping trampoline\n";
      continue;
    }
    std::optional<uint32_t> OriginalInstPrefLines =
        Elf.getKernelDescriptorInstPrefSize(KD.KernelName, TargetCpu);
    if (!OriginalInstPrefLines)
      return std::nullopt;
    uint32_t StubInstPrefLines =
        std::min(*OriginalInstPrefLines, KernelEntryStubInstPrefLines);
    MaxStubInstPrefLines = std::max(MaxStubInstPrefLines, StubInstPrefLines);
    Work.push_back({KD, StubInstPrefLines});
  }
  if (Work.empty())
    return 0;

  uint64_t AppendOffset = 0;
  for (const Trampoline &T : Growth)
    AppendOffset += T.Bytes.size();

  std::optional<uint64_t> PoolVAddrOr = Elf.trampolinePoolVAddr();
  if (!PoolVAddrOr)
    return std::nullopt;
  const uint64_t PoolVAddr = *PoolVAddrOr;

  std::optional<uint64_t> StubPoolBaseVAddr =
      checkedAddUint64(PoolVAddr, AppendOffset, "fast entry stub-pool base");
  if (!StubPoolBaseVAddr)
    return std::nullopt;
  std::optional<uint64_t> AlignedBase =
      checkedAlignTo(*StubPoolBaseVAddr, KernelEntryStubStride,
                     "fast entry trampoline aligned stub-pool base");
  if (!AlignedBase)
    return std::nullopt;
  const uint64_t StubStart = *AlignedBase - PoolVAddr;

  std::vector<Trampoline> LocalGrowth;
  std::vector<KernelEntryTrampolineFixup> LocalFixups;
  if (!appendPaddingFast(LocalGrowth, StubStart - AppendOffset))
    return std::nullopt;
  AppendOffset = StubStart;

  for (const WorkItem &Item : Work) {
    const KernelDescriptorInfo &KD = Item.KD;
    std::optional<uint64_t> StubVAddr = checkedAddUint64(
        PoolVAddr, AppendOffset, "fast entry trampoline vaddr");
    if (!StubVAddr)
      return std::nullopt;
    std::optional<unsigned> ScratchSgpr =
        allocateEntryStubScratchSgprsFast(Elf, KD, MaxSgprs);
    if (!ScratchSgpr)
      return std::nullopt;
    std::optional<uint64_t> Entry = entryVAddr(KD);
    if (!Entry)
      return std::nullopt;

    Trampoline T;
    T.Bytes = buildKernelEntryTrampolineFast(*StubVAddr, *Entry, *ScratchSgpr);
    LocalGrowth.push_back(std::move(T));

    // Per-kernel scratch pair s[*ScratchSgpr:*ScratchSgpr+1]: bump the
    // descriptor SGPR reservation (SkipSgprReservation=false) so the shared
    // rewriteKernelEntryDescriptorOffsets records the pair, exactly like the MC
    // path.
    LocalFixups.push_back({KD.KernelName, AppendOffset, *ScratchSgpr + 2,
                           Item.StubInstPrefLines,
                           /*SkipSgprReservation=*/false});

    std::optional<uint64_t> NewAppendOffset = checkedAddUint64(
        AppendOffset, KernelEntryStubStride, "fast entry append offset");
    if (!NewAppendOffset)
      return std::nullopt;
    AppendOffset = *NewAppendOffset;
  }

  // Prefetch guard sized like the MC path (shared helper).
  const uint64_t GuardBytes =
      computeKernelEntryPrefetchGuardBytes(MaxStubInstPrefLines);
  if (GuardBytes != 0) {
    Trampoline Guard;
    Guard.Bytes.reserve(GuardBytes);
    for (uint64_t Off = 0; Off < GuardBytes; Off += sizeof(SCodeEnd))
      Guard.Bytes.append(SCodeEnd, SCodeEnd + sizeof(SCodeEnd));
    LocalGrowth.push_back(std::move(Guard));
  }

  if (LocalFixups.empty())
    return 0;
  if (LocalFixups.size() > std::numeric_limits<uint32_t>::max()) {
    log() << "hotswap: error: fast kernel-entry trampoline count "
          << LocalFixups.size() << " exceeds uint32_t.\n";
    return std::nullopt;
  }

  for (Trampoline &T : LocalGrowth)
    Growth.push_back(std::move(T));
  OutFixups.insert(OutFixups.end(), LocalFixups.begin(), LocalFixups.end());

  log() << "hotswap: fast: installed " << LocalFixups.size()
        << " kernel-entry trampoline" << (LocalFixups.size() == 1 ? "" : "s")
        << " (no-disasm, per-kernel scratch SGPR) with " << GuardBytes
        << " prefetch guard bytes\n";
  return static_cast<uint32_t>(LocalFixups.size());
}

} // namespace hotswap
} // namespace COMGR
