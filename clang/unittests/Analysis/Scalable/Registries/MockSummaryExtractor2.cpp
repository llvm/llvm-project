//===- MockSummaryExtractor2.cpp ------------------------------------------===//
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
class MockSummaryExtractor2 : public TUSummaryExtractor {
public:
  MockSummaryExtractor2(TUSummaryBuilder &Builder)
      : TUSummaryExtractor(Builder) {
    getFakeBuilder().sendMessage(
        "MockSummaryExtractor2 constructor was invoked");
  }

  ~MockSummaryExtractor2() {
    getFakeBuilder().sendMessage(
        "MockSummaryExtractor2 destructor was invoked");
  }

  void HandleTranslationUnit(ASTContext &Ctx) override {
    getFakeBuilder().sendMessage(
        "MockSummaryExtractor2 HandleTranslationUnit was invoked");
  }

  MockTUSummaryBuilder &getFakeBuilder() {
    return static_cast<MockTUSummaryBuilder &>(SummaryBuilder);
  }
};

static TUSummaryExtractorRegistry::Add<MockSummaryExtractor2>
    RegisterExtractor("MockSummaryExtractor2", "Mock summary extractor 2");

} // namespace
