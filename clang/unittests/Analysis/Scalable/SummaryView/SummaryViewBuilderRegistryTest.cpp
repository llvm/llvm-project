//===- SummaryViewBuilderRegistryTest.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/SummaryView/SummaryViewBuilderRegistry.h"
#include "llvm/ADT/StringRef.h"
#include "gtest/gtest.h"
#include <memory>
#include <set>

using namespace clang;
using namespace ssaf;

namespace {

class SummaryViewBuilderRegistryTest : public ::testing::Test {};

TEST_F(SummaryViewBuilderRegistryTest, isSummaryViewBuilderRegistered) {
  EXPECT_FALSE(isSummaryViewBuilderRegistered("Non-existent-builder"));
  EXPECT_TRUE(isSummaryViewBuilderRegistered("Analysis1"));
  EXPECT_TRUE(isSummaryViewBuilderRegistered("Analysis2"));
  EXPECT_TRUE(isSummaryViewBuilderRegistered("Analysis4"));
}

TEST_F(SummaryViewBuilderRegistryTest, EnumeratingRegistryEntries) {
  std::set<llvm::StringRef> ActualNames;
  for (const auto &Entry : SummaryViewBuilderRegistry::entries()) {
    bool Inserted = ActualNames.insert(Entry.getName()).second;
    EXPECT_TRUE(Inserted);
  }

  EXPECT_EQ(ActualNames,
            (std::set<llvm::StringRef>{"Analysis1", "Analysis2", "Analysis4"}));
}

TEST_F(SummaryViewBuilderRegistryTest, InstantiatingBuilder_Analysis1) {
  std::unique_ptr<SummaryViewBuilderBase> B;
  for (const auto &Entry : SummaryViewBuilderRegistry::entries()) {
    if (Entry.getName() == "Analysis1")
      B = Entry.instantiate();
  }

  ASSERT_NE(B, nullptr);
  EXPECT_EQ(B->summaryName(), SummaryName("Analysis1"));
}

TEST_F(SummaryViewBuilderRegistryTest, InstantiatingBuilder_Analysis2) {
  std::unique_ptr<SummaryViewBuilderBase> B;
  for (const auto &Entry : SummaryViewBuilderRegistry::entries()) {
    if (Entry.getName() == "Analysis2")
      B = Entry.instantiate();
  }

  ASSERT_NE(B, nullptr);
  EXPECT_EQ(B->summaryName(), SummaryName("Analysis2"));
}

} // namespace
