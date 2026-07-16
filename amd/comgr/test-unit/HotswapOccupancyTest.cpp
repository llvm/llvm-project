//===- HotswapOccupancyTest.cpp - HotSwap capacity policy tests ----------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"
#include "comgr-test-elf-utils.h"
#include "gtest/gtest.h"

using namespace COMGR::hotswap;

TEST(HotswapOccupancy, LoadsGfx1250LimitsFromComgrIsaMetadata) {
  std::optional<SubtargetOccupancyLimits> Limits =
      getSubtargetOccupancyLimits("gfx1250");
  ASSERT_TRUE(Limits.has_value());
  EXPECT_EQ(Limits->EUsPerCU, 4u);
  EXPECT_EQ(Limits->MaxWavesPerCU, 40u);
  EXPECT_EQ(Limits->MaxFlatWorkgroupSize, 1024u);
  EXPECT_EQ(Limits->VgprAllocGranule, 16u);
  EXPECT_EQ(Limits->TotalNumVgprs, 1024u);
  EXPECT_TRUE(Limits->Wave64HalvesVgprCapacity);
}

TEST(HotswapOccupancy, PreservesExactWave32WorkgroupBoundary) {
  const SubtargetOccupancyLimits Limits{4, 40, 1024, 16, 1024, true};
  std::optional<WorkgroupCapacity> Capacity =
      computeWorkgroupCapacity(128, 1024, 32, Limits);
  ASSERT_TRUE(Capacity.has_value());
  EXPECT_EQ(Capacity->RequiredWavesPerEU, 8u);
  EXPECT_EQ(Capacity->AchievableWavesPerEU, 8u);
  EXPECT_EQ(decideVgprBump(PatchRequirement::Optional, *Capacity),
            VgprBumpDecision::Apply);
  EXPECT_EQ(decideVgprBump(PatchRequirement::Required, *Capacity),
            VgprBumpDecision::Apply);
}

TEST(HotswapOccupancy, DetectsOneVgprAcrossGranuleBoundary) {
  const SubtargetOccupancyLimits Limits{4, 40, 1024, 16, 1024, true};
  std::optional<WorkgroupCapacity> Capacity =
      computeWorkgroupCapacity(129, 1024, 32, Limits);
  ASSERT_TRUE(Capacity.has_value());
  EXPECT_EQ(Capacity->RequiredWavesPerEU, 8u);
  EXPECT_EQ(Capacity->AchievableWavesPerEU, 7u);
  EXPECT_EQ(decideVgprBump(PatchRequirement::Optional, *Capacity),
            VgprBumpDecision::Decline);
  EXPECT_EQ(decideVgprBump(PatchRequirement::Required, *Capacity),
            VgprBumpDecision::Fail);
}

TEST(HotswapOccupancy, HandlesWave64Workgroups) {
  const SubtargetOccupancyLimits Limits{4, 40, 1024, 16, 1024, true};
  std::optional<WorkgroupCapacity> Capacity =
      computeWorkgroupCapacity(128, 1024, 64, Limits);
  ASSERT_TRUE(Capacity.has_value());
  EXPECT_EQ(Capacity->RequiredWavesPerEU, 4u);
  EXPECT_EQ(Capacity->AchievableWavesPerEU, 4u);

  std::optional<WorkgroupCapacity> TooMany =
      computeWorkgroupCapacity(129, 1024, 64, Limits);
  ASSERT_TRUE(TooMany.has_value());
  EXPECT_EQ(TooMany->RequiredWavesPerEU, 4u);
  EXPECT_EQ(TooMany->AchievableWavesPerEU, 3u);
}

TEST(HotswapOccupancy, RejectsInvalidMetadata) {
  const SubtargetOccupancyLimits Limits{4, 40, 1024, 16, 1024, true};
  EXPECT_EQ(computeWorkgroupCapacity(128, 0, 32, Limits), std::nullopt);
  EXPECT_EQ(computeWorkgroupCapacity(128, 2048, 32, Limits), std::nullopt);
  EXPECT_EQ(computeWorkgroupCapacity(128, 1024, 0, Limits), std::nullopt);
  EXPECT_EQ(getSubtargetOccupancyLimits("not-a-gpu"), std::nullopt);
}

TEST(HotswapOccupancy, RejectsPatchOutsideKnownKernelWithZeroReportedGrowth) {
  const std::vector<uint8_t> Text(4);
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(Text);
  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  RewriteConfig Config;
  LLVMState LS;
  std::vector<InternalDecodedInst> Decoded;
  std::vector<Trampoline> Trampolines;
  std::vector<NopSled> Sleds;
  LivenessInfo Liveness;
  llvm::StringMap<KernelPatchStats> KernelStats;
  std::vector<ScratchPatchInfo> ScratchPatches;
  DirectControlFlowInfo ControlFlow;
  PatchContext Ctx{Config,
                   Decoded,
                   ViewOrErr->textData(),
                   ViewOrErr->textSize(),
                   /*PoolBaseOffset=*/0,
                   LS,
                   Trampolines,
                   Sleds,
                   *ViewOrErr,
                   Liveness,
                   KernelStats,
                   ScratchPatches,
                   ControlFlow};

  EXPECT_EQ(checkKernelVgprBump(Ctx, /*KernelName=*/{}, /*ExtraVgprs=*/0,
                                PatchRequirement::Optional),
            VgprBumpDecision::Decline);
  EXPECT_FALSE(Ctx.RequiredPatchFailed);

  EXPECT_EQ(checkKernelVgprBump(Ctx, /*KernelName=*/{}, /*ExtraVgprs=*/0,
                                PatchRequirement::Required),
            VgprBumpDecision::Fail);
  EXPECT_TRUE(Ctx.RequiredPatchFailed);
}
