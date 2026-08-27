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

static OffloadArch parse(llvm::StringRef S) { return StringToOffloadArch(S); }

TEST(OffloadArchTest, TargetArchClassification) {
  OffloadArch NV = parse("sm_120a");
  EXPECT_TRUE(parse("sm_20").isNVPTX());
  EXPECT_TRUE(NV.isNVPTX());
  EXPECT_FALSE(NV.isAMDGPU());

  EXPECT_TRUE(parse("gfx600").isAMDGPU());
  EXPECT_TRUE(parse("gfx1201").isAMDGPU());
  EXPECT_TRUE(parse("gfx12-generic").isAMDGPU());
  EXPECT_FALSE(parse("gfx600").isNVPTX());

  OffloadArch SPIRV = parse("amdgcnspirv");
  EXPECT_FALSE(SPIRV.isAMDGPU());
  EXPECT_TRUE(SPIRV.isSPIRV());

  OffloadArch IntelCPU = parse("graniterapids");
  EXPECT_FALSE(IntelCPU.isAMDGPU());
  EXPECT_FALSE(IntelCPU.isSPIRV());
  EXPECT_TRUE(IntelCPU.isIntel());
  EXPECT_TRUE(IntelCPU.isIntelCPU());
  EXPECT_FALSE(IntelCPU.isIntelGPU());

  OffloadArch IntelGPU = parse("bmg_g21");
  EXPECT_TRUE(IntelGPU.isIntel());
  EXPECT_FALSE(IntelGPU.isIntelCPU());
  EXPECT_TRUE(IntelGPU.isIntelGPU());

  OffloadArch Generic = parse("generic");
  EXPECT_FALSE(Generic.isNVPTX());
  EXPECT_FALSE(Generic.isAMDGPU());
  EXPECT_FALSE(Generic.isSPIRV());
  EXPECT_FALSE(Generic.isIntel());
}

TEST(OffloadArchTest, Unknown) {
  EXPECT_TRUE(parse("not-a-real-arch").isUnknown());
  EXPECT_TRUE(parse("").isUnused());
}

// Names must round-trip through parse -> string.
TEST(OffloadArchTest, RoundTrip) {
  for (const char *Name :
       {"sm_52", "sm_90a", "gfx906", "gfx1201", "gfx12-generic", "amdgcnspirv",
        "graniterapids", "bmg_g21", "generic"}) {
    OffloadArch A = parse(Name);
    EXPECT_FALSE(A.isUnknown()) << Name;
    EXPECT_STREQ(OffloadArchToString(A), Name);
  }
}

TEST(OffloadArchTest, Defaults) {
  EXPECT_STREQ(OffloadArchToString(OffloadArch::CudaDefault()), "sm_52");
  EXPECT_STREQ(OffloadArchToString(OffloadArch::HIPDefault()), "gfx906");
}

// Every AMDGPU gfx OffloadArch must round-trip through its subarch and
// back. This guards against adding a GPU to the TargetParser data but
// forgetting to update either getOffloadArchSubArch or the
// getSubArchOffloadArch table (the two are position-sensitive and easy to leave
// out of sync).
TEST(OffloadArchTest, AMDGPUSubArchRoundTrip) {
  for (int I = static_cast<int>(llvm::Triple::FirstAMDGPUSubArch);
       I <= static_cast<int>(llvm::Triple::LastAMDGPUSubArch); ++I) {
    auto SubArch = static_cast<llvm::Triple::SubArchType>(I);

    // Some subarch slots are gaps with no associated GPU; skip them.
    OffloadArch Arch = getSubArchOffloadArch(SubArch);
    if (Arch.isUnknown())
      continue;

    EXPECT_TRUE(Arch.isAMDGPU());
    EXPECT_EQ(getOffloadArchSubArch(Arch), SubArch)
        << "subarch round-trip failed for " << OffloadArchToString(Arch);
  }

  EXPECT_EQ(getOffloadArchSubArch(OffloadArch::getSPIRV()),
            llvm::Triple::NoSubArch);
}
