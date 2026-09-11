//===- unittests/Basic/TargetIDTest.cpp - Test TargetID -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/TargetParser/AMDGPUTargetParser.h"
#include "llvm/TargetParser/Triple.h"
#include "gtest/gtest.h"

namespace {
static std::string canonicalTargetID(const llvm::Triple &T, llvm::StringRef CPU,
                                     llvm::StringRef Features) {
  return llvm::AMDGPU::TargetID::createFromSubtargetFeatures(T, CPU, Features)
      .getCanonicalTargetIDString();
}

TEST(TargetIDTest, HIPBundleTargetIDPreservesXnack) {
  llvm::Triple T("amdgcn-amd-amdhsa");
  // An unrelated feature must not leak into the target ID.
  EXPECT_EQ(canonicalTargetID(T, "gfx90a", "+xnack,+wavefrontsize64"),
            "gfx90a:xnack+");
}

TEST(TargetIDTest, HIPBundleTargetIDPreservesDisabledFeature) {
  llvm::Triple T("amdgcn-amd-amdhsa");
  EXPECT_EQ(canonicalTargetID(T, "gfx90a", "-xnack"), "gfx90a:xnack-");
}

TEST(TargetIDTest, HIPBundleTargetIDWithoutFeatures) {
  llvm::Triple T("amdgcn-amd-amdhsa");
  EXPECT_EQ(canonicalTargetID(T, "gfx90a", ""), "gfx90a");
}

} // namespace
