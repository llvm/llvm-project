//===- SummaryExtractorRegistryTest.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MockTUSummaryBuilder.h"
#include "clang/Analysis/Scalable/TUSummary/ExtractorRegistry.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/StringRef.h"
#include "gtest/gtest.h"
#include <memory>

using namespace clang;
using namespace ssaf;

namespace {

TEST(SummaryExtractorRegistryTest, isTUSummaryExtractorRegistered) {
  EXPECT_FALSE(isTUSummaryExtractorRegistered("Non-existent-extractor"));
  EXPECT_TRUE(isTUSummaryExtractorRegistered("MockSummaryExtractor1"));
  EXPECT_TRUE(isTUSummaryExtractorRegistered("MockSummaryExtractor2"));
}

TEST(SummaryExtractorRegistryTest, EnumeratingRegistryEntries) {
  std::set<llvm::StringRef> ActualNames;
  for (const auto &Entry : TUSummaryExtractorRegistry::entries()) {
    bool Inserted = ActualNames.insert(Entry.getName()).second;
    EXPECT_TRUE(Inserted);
  }

  EXPECT_EQ(ActualNames, (std::set<llvm::StringRef>{
                             "MockSummaryExtractor1",
                             "MockSummaryExtractor2",
                         }));
}

TEST(SummaryExtractorRegistryTest, InstantiatingExtractor1) {
  MockTUSummaryBuilder FakeBuilder;
  {
    auto Consumer =
        makeTUSummaryExtractor("MockSummaryExtractor1", FakeBuilder);
    EXPECT_TRUE(Consumer);
    EXPECT_EQ(FakeBuilder.consumeMessages(),
              "MockSummaryExtractor1 constructor was invoked\n");
  }
  EXPECT_EQ(FakeBuilder.consumeMessages(),
            "MockSummaryExtractor1 destructor was invoked\n");
}

TEST(SummaryExtractorRegistryTest, InstantiatingExtractor2) {
  MockTUSummaryBuilder FakeBuilder;
  {
    auto Consumer =
        makeTUSummaryExtractor("MockSummaryExtractor2", FakeBuilder);
    EXPECT_TRUE(Consumer);
    EXPECT_EQ(FakeBuilder.consumeMessages(),
              "MockSummaryExtractor2 constructor was invoked\n");
  }
  EXPECT_EQ(FakeBuilder.consumeMessages(),
            "MockSummaryExtractor2 destructor was invoked\n");
}

TEST(SummaryExtractorRegistryTest, InvokingExtractors) {
  MockTUSummaryBuilder FakeBuilder;
  std::vector<std::unique_ptr<ASTConsumer>> Consumers;
  for (std::string Name : {"MockSummaryExtractor1", "MockSummaryExtractor2"}) {
    auto Consumer = makeTUSummaryExtractor(Name, FakeBuilder);
    ASSERT_TRUE(Consumer);
    Consumers.push_back(std::move(Consumer));
  }
  EXPECT_EQ(FakeBuilder.consumeMessages(),
            "MockSummaryExtractor1 constructor was invoked\n"
            "MockSummaryExtractor2 constructor was invoked\n");

  {
    MultiplexConsumer Multiplexer(std::move(Consumers));
    auto AST = tooling::buildASTFromCode(R"cpp(int x = 42;)cpp");
    ASSERT_TRUE(AST);

    Multiplexer.HandleTranslationUnit(AST->getASTContext());
    EXPECT_EQ(FakeBuilder.consumeMessages(),
              "MockSummaryExtractor1 HandleTranslationUnit was invoked\n"
              "MockSummaryExtractor2 HandleTranslationUnit was invoked\n");
  }

  std::string OwnedMessages = FakeBuilder.consumeMessages();
  StringRef View = OwnedMessages;
  EXPECT_EQ(View.count('\n'), 2u);

  // The destruction order is indeterminate.
  EXPECT_TRUE(View.contains("MockSummaryExtractor1 destructor was invoked\n"));
  EXPECT_TRUE(View.contains("MockSummaryExtractor2 destructor was invoked\n"));
}

} // namespace
