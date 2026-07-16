//===- comgr-hotswap-occupancy.cpp - VGPR capacity policy -----------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Checks proposed hotswap VGPR growth against the waves needed to admit one
/// maximum-size workgroup. Optional transformations are declined before they
/// emit bytes when the growth would violate that invariant; required
/// transformations fail the rewrite.
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"

#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <limits>

using namespace llvm;

namespace COMGR {
namespace hotswap {

std::optional<SubtargetOccupancyLimits>
getSubtargetOccupancyLimits(StringRef Processor) {
#define HANDLE_ISA(TARGET_TRIPLE, PROCESSOR, SRAMECC_SUPPORTED,                \
                   XNACK_ON_OFF_MODES, ELF_MACHINE, TRAP_HANDLER_ENABLED,      \
                   IMAGE_SUPPORT, LDS_SIZE, LDS_BANK_COUNT, EUS_PER_CU,        \
                   MAX_WAVES_PER_CU, MAX_FLAT_WORK_GROUP_SIZE,                 \
                   SGPR_ALLOC_GRANULE, TOTAL_NUM_SGPRS, ADDRESSABLE_NUM_SGPRS, \
                   VGPR_ALLOC_GRANULE, TOTAL_NUM_VGPRS, ADDRESSABLE_NUM_VGPRS) \
  if (Processor == PROCESSOR)                                                  \
    return SubtargetOccupancyLimits{EUS_PER_CU,                                \
                                    MAX_WAVES_PER_CU,                          \
                                    MAX_FLAT_WORK_GROUP_SIZE,                  \
                                    VGPR_ALLOC_GRANULE,                        \
                                    TOTAL_NUM_VGPRS,                           \
                                    StringRef(PROCESSOR).starts_with("gfx1")};
#include "comgr-isa-metadata.def"
#undef HANDLE_ISA

  log() << "hotswap: error: no occupancy limits for processor '" << Processor
        << "'.\n";
  return std::nullopt;
}

std::optional<WorkgroupCapacity>
computeWorkgroupCapacity(unsigned Vgprs, unsigned MaxFlatWorkgroupSize,
                         unsigned WavefrontSize,
                         const SubtargetOccupancyLimits &Limits) {
  if (Vgprs == 0 || MaxFlatWorkgroupSize == 0 ||
      (WavefrontSize != 32 && WavefrontSize != 64) || Limits.EUsPerCU == 0 ||
      Limits.MaxWavesPerCU == 0 || Limits.MaxFlatWorkgroupSize == 0 ||
      Limits.VgprAllocGranule == 0 || Limits.TotalNumVgprs == 0 ||
      MaxFlatWorkgroupSize > Limits.MaxFlatWorkgroupSize) {
    log() << "hotswap: error: invalid inputs to computeWorkgroupCapacity: "
          << "vgprs=" << Vgprs
          << ", max_flat_workgroup_size=" << MaxFlatWorkgroupSize
          << ", wavefront_size=" << WavefrontSize
          << ", eus_per_cu=" << Limits.EUsPerCU
          << ", max_waves_per_cu=" << Limits.MaxWavesPerCU
          << ", target_max_flat_workgroup_size=" << Limits.MaxFlatWorkgroupSize
          << ", vgpr_granule=" << Limits.VgprAllocGranule
          << ", total_vgprs=" << Limits.TotalNumVgprs << ".\n";
    return std::nullopt;
  }

  unsigned VgprAllocGranule = Limits.VgprAllocGranule;
  unsigned TotalNumVgprs = Limits.TotalNumVgprs;
  if (WavefrontSize == 64 && Limits.Wave64HalvesVgprCapacity) {
    if ((VgprAllocGranule % 2) != 0 || (TotalNumVgprs % 2) != 0) {
      log() << "hotswap: error: wave64 cannot halve odd VGPR limits.\n";
      return std::nullopt;
    }
    VgprAllocGranule /= 2;
    TotalNumVgprs /= 2;
  }

  uint64_t RoundedVgprs =
      ((static_cast<uint64_t>(Vgprs) + VgprAllocGranule - 1) /
       VgprAllocGranule) *
      VgprAllocGranule;
  if (RoundedVgprs == 0 ||
      RoundedVgprs > std::numeric_limits<unsigned>::max()) {
    log() << "hotswap: error: rounded VGPR count " << RoundedVgprs
          << " is outside unsigned range.\n";
    return std::nullopt;
  }

  unsigned WavesPerWorkgroup = 1 + (MaxFlatWorkgroupSize - 1) / WavefrontSize;
  unsigned RequiredWavesPerEU = 1 + (WavesPerWorkgroup - 1) / Limits.EUsPerCU;
  unsigned HardwareWavesPerEU = Limits.MaxWavesPerCU / Limits.EUsPerCU;
  if (HardwareWavesPerEU == 0) {
    log()
        << "hotswap: error: target has fewer maximum waves per CU than EUs.\n";
    return std::nullopt;
  }

  unsigned VgprWavesPerEU = static_cast<unsigned>(
      TotalNumVgprs / static_cast<unsigned>(RoundedVgprs));
  return WorkgroupCapacity{
      RequiredWavesPerEU,
      std::min(VgprWavesPerEU, HardwareWavesPerEU),
  };
}

VgprBumpDecision decideVgprBump(PatchRequirement Requirement,
                                const WorkgroupCapacity &Capacity) {
  if (Capacity.AchievableWavesPerEU >= Capacity.RequiredWavesPerEU)
    return VgprBumpDecision::Apply;
  return Requirement == PatchRequirement::Required ? VgprBumpDecision::Fail
                                                   : VgprBumpDecision::Decline;
}

static VgprBumpDecision failOrDeclineVgprBump(PatchContext &Ctx,
                                              PatchRequirement Requirement) {
  if (Requirement == PatchRequirement::Required) {
    Ctx.RequiredPatchFailed = true;
    return VgprBumpDecision::Fail;
  }
  return VgprBumpDecision::Decline;
}

unsigned getKernelVgprGranuleSize(PatchContext &Ctx, StringRef KernelName) {
  StringMap<unsigned>::const_iterator Cached =
      Ctx.KernelVgprGranuleCache.find(KernelName);
  if (Cached != Ctx.KernelVgprGranuleCache.end())
    return Cached->second;

  unsigned Granule = Ctx.Config.VgprGranuleSize;
  std::optional<unsigned> WavefrontSize =
      Ctx.Elf.getKernelWavefrontSize(KernelName);
  std::optional<SubtargetOccupancyLimits> Limits =
      getSubtargetOccupancyLimits(Ctx.Config.TargetCpu);
  if (WavefrontSize && Limits && *WavefrontSize == 64 &&
      Limits->Wave64HalvesVgprCapacity && (Limits->VgprAllocGranule % 2) == 0)
    Granule = Limits->VgprAllocGranule / 2;
  else if (WavefrontSize && Limits && *WavefrontSize == 32)
    Granule = Limits->VgprAllocGranule;

  Ctx.KernelVgprGranuleCache.try_emplace(KernelName, Granule);
  return Granule;
}

VgprBumpDecision checkKernelVgprBump(PatchContext &Ctx, StringRef KernelName,
                                     unsigned ExtraVgprs,
                                     PatchRequirement Requirement) {
  if (KernelName.empty()) {
    // TODO: Build a device-function call graph and propagate the transformed
    // function's VGPR requirement to every reachable kernel. Until then, an
    // allocator result relative to a fallback register count cannot prove that
    // any caller's descriptor covers the selected scratch registers.
    log() << "hotswap: "
          << (Requirement == PatchRequirement::Required ? "error: " : "")
          << "cannot verify VGPR capacity for a patch site outside a known "
             "kernel because its calling kernels are unknown; "
          << (Requirement == PatchRequirement::Required
                  ? "failing required patch"
                  : "declining optional patch")
          << ".\n";
    return failOrDeclineVgprBump(Ctx, Requirement);
  }

  if (ExtraVgprs == 0)
    return VgprBumpDecision::Apply;

  unsigned AggregateExtraVgprs = ExtraVgprs;
  StringMap<KernelPatchStats>::const_iterator Stats =
      Ctx.KernelStats.find(KernelName);
  if (Stats != Ctx.KernelStats.end())
    AggregateExtraVgprs =
        std::max(AggregateExtraVgprs, Stats->second.ExtraVgprs);

  unsigned VgprGranuleSize = getKernelVgprGranuleSize(Ctx, KernelName);
  std::optional<unsigned> CurrentVgprs =
      Ctx.Elf.getKernelVgprCount(KernelName, VgprGranuleSize);
  if (!CurrentVgprs ||
      AggregateExtraVgprs >
          std::numeric_limits<unsigned>::max() - *CurrentVgprs) {
    log() << "hotswap: error: cannot compute proposed VGPR count for kernel '"
          << KernelName << "'.\n";
    return failOrDeclineVgprBump(Ctx, Requirement);
  }
  unsigned ProposedVgprs = *CurrentVgprs + AggregateExtraVgprs;

  std::optional<SubtargetOccupancyLimits> Limits =
      getSubtargetOccupancyLimits(Ctx.Config.TargetCpu);
  if (!Limits)
    return failOrDeclineVgprBump(Ctx, Requirement);

  StringMap<std::optional<KernelWorkgroupMetadata>>::iterator Cached =
      Ctx.WorkgroupMetadataCache.find(KernelName);
  if (Cached == Ctx.WorkgroupMetadataCache.end()) {
    std::optional<unsigned> MaxFlatWorkgroupSize =
        Ctx.Elf.getKernelMaxFlatWorkgroupSize(KernelName);
    std::optional<unsigned> WavefrontSize =
        Ctx.Elf.getKernelWavefrontSize(KernelName);
    std::optional<KernelWorkgroupMetadata> Metadata;
    if (MaxFlatWorkgroupSize && WavefrontSize)
      Metadata = KernelWorkgroupMetadata{*MaxFlatWorkgroupSize, *WavefrontSize};
    Cached = Ctx.WorkgroupMetadataCache.try_emplace(KernelName, Metadata).first;
  }

  if (!Cached->second) {
    log() << "hotswap: "
          << (Requirement == PatchRequirement::Required ? "error: " : "")
          << "cannot verify VGPR capacity for kernel '" << KernelName
          << "' because .max_flat_workgroup_size or .wavefront_size metadata "
             "is unavailable; "
          << (Requirement == PatchRequirement::Required
                  ? "failing required patch"
                  : "declining optional patch")
          << ".\n";
    return failOrDeclineVgprBump(Ctx, Requirement);
  }

  std::optional<WorkgroupCapacity> Capacity = computeWorkgroupCapacity(
      ProposedVgprs, Cached->second->MaxFlatWorkgroupSize,
      Cached->second->WavefrontSize, *Limits);
  if (!Capacity)
    return failOrDeclineVgprBump(Ctx, Requirement);

  VgprBumpDecision Decision = decideVgprBump(Requirement, *Capacity);
  if (Decision == VgprBumpDecision::Apply)
    return Decision;
  if (Decision == VgprBumpDecision::Fail)
    Ctx.RequiredPatchFailed = true;

  log() << "hotswap: " << (Decision == VgprBumpDecision::Fail ? "error: " : "")
        << (Decision == VgprBumpDecision::Fail ? "required" : "optional")
        << " patch for kernel '" << KernelName << "' would grow VGPRs from "
        << *CurrentVgprs << " to " << ProposedVgprs
        << " and reduce capacity to " << Capacity->AchievableWavesPerEU
        << " waves/EU, below the " << Capacity->RequiredWavesPerEU
        << " waves/EU needed for one maximum-size workgroup; "
        << (Decision == VgprBumpDecision::Fail ? "failing rewrite"
                                               : "declining patch")
        << ".\n";
  return Decision;
}

} // namespace hotswap
} // namespace COMGR
