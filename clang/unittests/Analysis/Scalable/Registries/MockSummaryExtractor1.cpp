//===- MockSummaryExtractor1.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MockTUSummaryBuilder.h"
#include "clang/AST/ASTContext.h"
#include "clang/Analysis/Scalable/TUSummary/ExtractorRegistry.h"
#include "clang/Analysis/Scalable/TUSummary/TUSummaryBuilder.h"

using namespace clang;
using namespace ssaf;

namespace {
class MockSummaryExtractor1 : public TUSummaryExtractor {
public:
  MockSummaryExtractor1(TUSummaryBuilder &Builder)
      : TUSummaryExtractor(Builder) {
    getFakeBuilder().sendMessage(
        "MockSummaryExtractor1 constructor was invoked");
  }

  ~MockSummaryExtractor1() {
    getFakeBuilder().sendMessage(
        "MockSummaryExtractor1 destructor was invoked");
  }

  void HandleTranslationUnit(ASTContext &Ctx) override {
    getFakeBuilder().sendMessage(
        "MockSummaryExtractor1 HandleTranslationUnit was invoked");
  }

  MockTUSummaryBuilder &getFakeBuilder() {
    return static_cast<MockTUSummaryBuilder &>(SummaryBuilder);
  }
};

static TUSummaryExtractorRegistry::Add<MockSummaryExtractor1>
    RegisterExtractor("MockSummaryExtractor1", "Mock summary extractor 1");

} // namespace
