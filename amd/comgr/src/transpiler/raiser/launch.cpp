//===- launch.cpp - Transpiler launch requirements ---------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "transpiler/raiser/launch.h"

#include "transpiler/raiser/raise_failure.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/MathExtras.h"

#include <cassert>

using namespace llvm;

namespace COMGR::transpiler {

Expected<LaunchDimensions> KernelLaunchRequirements::project(
    StringRef KernelName, const LaunchDimensions &Source,
    ArrayRef<uint8_t> Kernarg, uint32_t DynamicLDSSize) const {
  assert(!KernelName.empty() && "launch requirements need a kernel name");
  auto Refuse = [&](const Twine &Detail) {
    return RaiseFailure::inKernel(RaiseFailureReason::UnsupportedLaunch,
                                  KernelName, Detail);
  };
  if (RequiredWorkgroupSize && Source.Workgroup != *RequiredWorkgroupSize)
    return Refuse(
        "workgroup does not match the source kernel's required dimensions");
  uint64_t Workitems = 1;
  for (size_t I = 0; I != Source.Grid.size(); ++I) {
    if (Source.Grid[I] == 0 || Source.Workgroup[I] == 0)
      return Refuse("launch dimensions must be nonzero");
    // Bound each factor before multiplying, including untrusted dimensions.
    if (Source.Workgroup[I] > MaxWorkgroupSize)
      return Refuse("workgroup exceeds the kernel's supported launch size");
    Workitems *= Source.Workgroup[I];
    if (Workitems > MaxWorkgroupSize)
      return Refuse("workgroup exceeds the kernel's supported launch size");
  }
  if (RequiredWorkgroupSize)
    for (unsigned I = 0; I != 3; ++I)
      if (Source.Grid[I] % Source.Workgroup[I] != 0)
        return Refuse("required workgroup dimensions need complete workgroups");
  if (DynamicLDSSizeArgOffset) {
    uint32_t Offset = *DynamicLDSSizeArgOffset;
    if (Offset > Kernarg.size() || Kernarg.size() - Offset < sizeof(uint32_t))
      return Refuse("kernarg is missing the dynamic LDS size");
    if (support::endian::read32le(Kernarg.data() + Offset) != DynamicLDSSize)
      return Refuse("kernarg dynamic LDS size does not match the allocation");
  }
  if (Mapping == Kind::Unchanged) {
    if (RequiresWholeSourceWaves && Workitems % SourceWaveSize != 0)
      return Refuse("launch requires whole source waves");
    if (RequiresWholeSourceWaves && SourceWaveSize > 1)
      for (unsigned I = 0; I != 3; ++I)
        if (Source.Grid[I] % Source.Workgroup[I] != 0)
          return Refuse("launch requires complete workgroups");
    return Source;
  }

  assert(SourceWaveSize > 0 && ReplicationFactor > 1 &&
         "invalid replicated launch requirements");
  if (Mapping == Kind::Replicated1D &&
      (Source.Workgroup[1] != 1 || Source.Workgroup[2] != 1))
    return Refuse("replicated dispatch requires a one-dimensional workgroup");
  if (RequiresWholeSourceWaves && Workitems % SourceWaveSize != 0)
    return Refuse("replicated dispatch requires whole source waves");
  for (unsigned I = 0; I != 3; ++I)
    if (Source.Grid[I] % Source.Workgroup[I] != 0)
      return Refuse("replicated dispatch requires complete workgroups");
  if (WorkgroupSizeArgOffsets) {
    for (unsigned I = 0; I != 3; ++I) {
      uint32_t Offset = (*WorkgroupSizeArgOffsets)[I];
      if (Offset > Kernarg.size() || Kernarg.size() - Offset < sizeof(uint16_t))
        return Refuse("kernarg is missing a logical workgroup size");
      if (support::endian::read16le(Kernarg.data() + Offset) !=
          Source.Workgroup[I])
        return Refuse(
            "kernarg workgroup size does not match the logical launch");
    }
  }

  uint64_t PhysicalWorkitems =
      alignTo(Workitems, SourceWaveSize) * ReplicationFactor;
  uint64_t GridX = Source.Grid[0] / Source.Workgroup[0] * PhysicalWorkitems;
  if (GridX > UINT32_MAX)
    return Refuse("replicated grid size overflows the dispatch packet");

  return LaunchDimensions{{static_cast<uint32_t>(GridX),
                           Source.Grid[1] / Source.Workgroup[1],
                           Source.Grid[2] / Source.Workgroup[2]},
                          {static_cast<uint32_t>(PhysicalWorkitems), 1, 1}};
}

} // namespace COMGR::transpiler
