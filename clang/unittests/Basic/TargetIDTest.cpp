//===- unittests/Basic/TargetIDTest.cpp - Test TargetID -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/TargetID.h"
#include "llvm/TargetParser/AMDGPUTargetParser.h"
#include "llvm/TargetParser/Triple.h"
#include "gtest/gtest.h"

namespace {

static std::string bundleEntryID(const llvm::Triple &T, llvm::StringRef CPU,
                                 llvm::StringRef Features,
                                 llvm::StringRef OffloadKind) {
  std::string TargetID =
      llvm::AMDGPU::TargetID::createFromSubtargetFeatures(T, CPU, Features)
          .getCanonicalFeatureString();
  return OffloadKind.str() + "-" + clang::normalizeForBundler(T, TargetID) +
         "-" + TargetID;
}

TEST(TargetIDTest, HIPBundleEntryPreservesXnack) {
  llvm::Triple T("amdgcn-amd-amdhsa");
  EXPECT_EQ(bundleEntryID(T, "gfx90a", "+xnack,+wavefrontsize64", "hipv4"),
            "hipv4-amdgcn-amd-amdhsa--gfx90a:xnack+");
}

TEST(TargetIDTest, HIPBundleEntryPreservesDisabledFeature) {
  llvm::Triple T("amdgcn-amd-amdhsa");
  EXPECT_EQ(bundleEntryID(T, "gfx90a", "-xnack", "hipv4"),
            "hipv4-amdgcn-amd-amdhsa--gfx90a:xnack-");
}

TEST(TargetIDTest, HIPBundleEntryWithoutFeatures) {
  llvm::Triple T("amdgcn-amd-amdhsa");
  EXPECT_EQ(bundleEntryID(T, "gfx90a", "", "hipv4"),
            "hipv4-amdgcn-amd-amdhsa--gfx90a");
}

TEST(TargetIDTest, HIPBundleEntryLegacyKind) {
  llvm::Triple T("amdgcn-amd-amdhsa");
  EXPECT_EQ(bundleEntryID(T, "gfx908", "+sramecc", "hip"),
            "hip-amdgcn-amd-amdhsa--gfx908:sramecc+");
}

} // namespace
