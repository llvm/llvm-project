//===- SourcePassAnalysisRegistryTest.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysisFramework/Core/SourcePassAnalysis/SourcePassAnalysisRegistry.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisName.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisResult.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/WPASuite.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <memory>

using namespace clang;
using namespace ssaf;

namespace {

// ---------------------------------------------------------------------------
// Dummy analysis result & source-pass analysis
// ---------------------------------------------------------------------------

static const AnalysisName
    DummyDependentAnalysisResultName("DummyDependentAnalysisResult");
static const AnalysisName
    DummySourcePassAnalysisName("DummySourcePassAnalysis");

// Dummy AnalysisResult that the dummy source pass analysis depends on:
class DummyWPAResult final : public AnalysisResult {
public:
  static AnalysisName analysisName() {
    return DummyDependentAnalysisResultName;
  }
};

// Dummy source pass analysis:
class DummySourcePassAnalysis final
    : public SourcePassAnalysis<void, DummyWPAResult> {
public:
  using SourcePassAnalysis::SourcePassAnalysis;

  static AnalysisName analysisName() { return DummySourcePassAnalysisName; }
};

// Registration — this TU is compiled directly into the test binary, so the
// static initializer is guaranteed to run.
static SourcePassAnalysisRegistry::Add<DummySourcePassAnalysis, void,
                                       DummyWPAResult>
    RegTestSPA("Test source-pass analysis");

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(SourcePassAnalysisRegistryTest, ContainsRegistered) {
  EXPECT_TRUE(
      SourcePassAnalysisRegistry::contains(DummySourcePassAnalysisName));
}

TEST(SourcePassAnalysisRegistryTest, DoesNotContainUnregistered) {
  EXPECT_FALSE(
      SourcePassAnalysisRegistry::contains(AnalysisName("NonExistent")));
}

TEST(SourcePassAnalysisRegistryTest, InstantiateRegistered) {
  auto Result = SourcePassAnalysisRegistry::instantiate(
      DummySourcePassAnalysis::analysisName(), nullptr);
  EXPECT_THAT_EXPECTED(Result, llvm::Succeeded());
}

TEST(SourcePassAnalysisRegistryTest, InstantiateUnregistered) {
  auto Result = SourcePassAnalysisRegistry::instantiate(
      AnalysisName("NonExistent"), nullptr);
  EXPECT_THAT_EXPECTED(
      Result, llvm::FailedWithMessage("no source-pass analysis registered for "
                                      "'AnalysisName(NonExistent)'"));
}

} // namespace
