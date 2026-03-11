//===- SummaryViewBuilderRegistryTest.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/SummaryView/SummaryViewBuilderRegistry.h"
#include "MockSummaryViewBuilders.h"
#include "llvm/ADT/StringRef.h"
#include "gtest/gtest.h"
#include <set>

using namespace clang;
using namespace ssaf;

namespace {

class SummaryViewBuilderRegistryTest : public ::testing::Test {
protected:
  void SetUp() override { clearMockBuilderLog(); }
};

TEST_F(SummaryViewBuilderRegistryTest, isSummaryViewBuilderRegistered) {
  EXPECT_FALSE(isSummaryViewBuilderRegistered("Non-existent-builder"));
  EXPECT_TRUE(isSummaryViewBuilderRegistered("Mock1"));
  EXPECT_TRUE(isSummaryViewBuilderRegistered("Mock2"));
}

TEST_F(SummaryViewBuilderRegistryTest, EnumeratingRegistryEntries) {
  std::set<llvm::StringRef> ActualNames;
  for (const auto &Entry : SummaryViewBuilderRegistry::entries()) {
    bool Inserted = ActualNames.insert(Entry.getName()).second;
    EXPECT_TRUE(Inserted);
  }

  EXPECT_EQ(ActualNames, (std::set<llvm::StringRef>{"Mock1", "Mock2"}));
}

TEST_F(SummaryViewBuilderRegistryTest, InstantiatingBuilder1) {
  {
    // Find Mock1 entry explicitly.
    std::unique_ptr<SummaryViewBuilderBase> B1;
    for (const auto &Entry : SummaryViewBuilderRegistry::entries()) {
      if (Entry.getName() == "Mock1") {
        B1 = Entry.instantiate();
      }
    }

    ASSERT_TRUE(B1);
    EXPECT_EQ(B1->summaryName(), SummaryName("Mock1"));
    EXPECT_TRUE(MockBuilderLog.find(
                    "MockSummaryViewBuilder1 constructor was invoked") !=
                std::string::npos);
  }
  EXPECT_TRUE(
      MockBuilderLog.find("MockSummaryViewBuilder1 destructor was invoked") !=
      std::string::npos);
}

TEST_F(SummaryViewBuilderRegistryTest, InstantiatingBuilder2) {
  {
    std::unique_ptr<SummaryViewBuilderBase> B2;
    for (const auto &Entry : SummaryViewBuilderRegistry::entries()) {
      if (Entry.getName() == "Mock2") {
        B2 = Entry.instantiate();
      }
    }

    ASSERT_TRUE(B2);
    EXPECT_EQ(B2->summaryName(), SummaryName("Mock2"));
    EXPECT_TRUE(MockBuilderLog.find(
                    "MockSummaryViewBuilder2 constructor was invoked") !=
                std::string::npos);
  }
  EXPECT_TRUE(
      MockBuilderLog.find("MockSummaryViewBuilder2 destructor was invoked") !=
      std::string::npos);
}

} // namespace
