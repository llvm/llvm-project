//===- unittests/Basic/TargetIDTest.cpp - Test TargetID -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/TargetID.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/TargetParser/Triple.h"
#include "gtest/gtest.h"

using namespace clang;

namespace {

static std::string canonicalTargetID(const llvm::Triple &T, llvm::StringRef CPU,
                                     const llvm::StringMap<bool> &FeatureMap) {
  llvm::StringMap<bool> IDFeatures;
  for (llvm::StringRef Feature : getAllPossibleTargetIDFeatures(T, CPU)) {
    auto It = FeatureMap.find(Feature);
    if (It != FeatureMap.end())
      IDFeatures[Feature] = It->second;
  }
  return getCanonicalTargetID(CPU, IDFeatures);
}

TEST(TargetIDTest, HIPBundleTargetIDPreservesXnack) {
  llvm::Triple T("amdgcn-amd-amdhsa");
  llvm::StringMap<bool> FeatureMap;
  FeatureMap["xnack"] = true;
  FeatureMap["wavefrontsize64"] = true;

  EXPECT_EQ(canonicalTargetID(T, "gfx90a", FeatureMap), "gfx90a:xnack+");
}

TEST(TargetIDTest, HIPBundleTargetIDPreservesDisabledFeature) {
  llvm::Triple T("amdgcn-amd-amdhsa");
  llvm::StringMap<bool> FeatureMap;
  FeatureMap["xnack"] = false;

  EXPECT_EQ(canonicalTargetID(T, "gfx90a", FeatureMap), "gfx90a:xnack-");
}

TEST(TargetIDTest, HIPBundleTargetIDWithoutFeatures) {
  llvm::Triple T("amdgcn-amd-amdhsa");
  llvm::StringMap<bool> FeatureMap;

  EXPECT_EQ(canonicalTargetID(T, "gfx90a", FeatureMap), "gfx90a");
}

} // namespace
