//===- unittests/Basic/OffloadArchTest.cpp - Test OffloadArch -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/OffloadArch.h"
#include "llvm/TargetParser/Triple.h"
#include "gtest/gtest.h"

using namespace clang;

TEST(OffloadArchTest, basic) {
  EXPECT_TRUE(IsNVIDIAOffloadArch(OffloadArch::SM_20));
  EXPECT_TRUE(IsNVIDIAOffloadArch(OffloadArch::SM_120a));
  EXPECT_FALSE(IsNVIDIAOffloadArch(OffloadArch::GFX600));

  EXPECT_FALSE(IsAMDOffloadArch(OffloadArch::SM_120a));
  EXPECT_TRUE(IsAMDOffloadArch(OffloadArch::GFX600));
  EXPECT_TRUE(IsAMDOffloadArch(OffloadArch::GFX1201));
  EXPECT_TRUE(IsAMDOffloadArch(OffloadArch::GFX12_GENERIC));
  EXPECT_TRUE(IsAMDOffloadArch(OffloadArch::AMDGCNSPIRV));
  EXPECT_FALSE(IsAMDOffloadArch(OffloadArch::GRANITERAPIDS));

  EXPECT_TRUE(IsIntelOffloadArch(OffloadArch::GRANITERAPIDS));
  EXPECT_TRUE(IsIntelCPUOffloadArch(OffloadArch::GRANITERAPIDS));
  EXPECT_FALSE(IsIntelGPUOffloadArch(OffloadArch::GRANITERAPIDS));
  EXPECT_TRUE(IsIntelOffloadArch(OffloadArch::BMG_G21));
  EXPECT_FALSE(IsIntelCPUOffloadArch(OffloadArch::BMG_G21));
  EXPECT_TRUE(IsIntelGPUOffloadArch(OffloadArch::BMG_G21));

  EXPECT_FALSE(IsNVIDIAOffloadArch(OffloadArch::Generic));
  EXPECT_FALSE(IsAMDOffloadArch(OffloadArch::Generic));
  EXPECT_FALSE(IsIntelOffloadArch(OffloadArch::Generic));
}

// Every AMDGPU gfx OffloadArch must round-trip through its subarch and
// back. This guards against adding a GPU to the OffloadArch enum but forgetting
// to update either getOffloadArchSubArch or the getSubArchOffloadArch table
// (the two are position-sensitive and easy to leave out of sync).
TEST(OffloadArchTest, AMDGPUSubArchRoundTrip) {
  for (int I = static_cast<int>(OffloadArch::GFX600);
       I < static_cast<int>(OffloadArch::AMDGCNSPIRV); ++I) {
    OffloadArch Arch = static_cast<OffloadArch>(I);
    EXPECT_TRUE(IsAMDOffloadArch(Arch));

    llvm::Triple::SubArchType SubArch = getOffloadArchSubArch(Arch);
    EXPECT_NE(SubArch, llvm::Triple::NoSubArch)
        << "missing subarch mapping for " << OffloadArchToString(Arch);
    EXPECT_EQ(getSubArchOffloadArch(SubArch), Arch)
        << "subarch round-trip failed for " << OffloadArchToString(Arch);
  }

  EXPECT_EQ(getOffloadArchSubArch(OffloadArch::AMDGCNSPIRV),
            llvm::Triple::NoSubArch);
}
