//===- unittests/Basic/TargetIDTest.cpp - Test TargetID -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/TargetID.h"
#include "llvm/TargetParser/Triple.h"
#include "gtest/gtest.h"

namespace {

static std::string bundleEntryID(const llvm::Triple &T, llvm::StringRef CPU,
                                 llvm::StringRef Features,
                                 llvm::StringRef OffloadKind) {
  llvm::SmallVector<llvm::StringRef, 4> FeatureList;
  Features.split(FeatureList, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  llvm::StringMap<bool> TargetIDFeatures;
  for (llvm::StringRef Name : clang::getAllPossibleTargetIDFeatures(T, CPU))
    for (llvm::StringRef F : FeatureList)
      if (F.drop_front() == Name)
        TargetIDFeatures[Name] = F.front() == '+';
  std::string TargetID = clang::getCanonicalTargetID(CPU, TargetIDFeatures);
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
