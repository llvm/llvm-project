//===- unittests/Analysis/IntervalPartitionTest.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/IntervalPartition.h"
#include "CFGBuildResult.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace analysis {
namespace {

TEST(BuildInterval, PartitionSimpleOneInterval) {

  const char *Code = R"(void f() {
                          int x = 3;
                          int y = 7;
                          x = y + x;
                        })";
  BuildResult Result = BuildCFG(Code);
  EXPECT_EQ(BuildResult::BuiltCFG, Result.getStatus());

  CFG *cfg = Result.getCFG();

  // Basic correctness checks.
  ASSERT_EQ(cfg->size(), 3u);

  auto &EntryBlock = cfg->getEntry();

  CFGInterval I = buildInterval(*cfg, EntryBlock);
  EXPECT_EQ(I.Blocks.size(), 3u);
}

TEST(BuildInterval, PartitionIfThenOneInterval) {

  const char *Code = R"(void f() {
                          int x = 3;
                          if (x > 3)
                            x = 2;
                          else
                            x = 7;
                          x = x + x;
                        })";
  BuildResult Result = BuildCFG(Code);
  EXPECT_EQ(BuildResult::BuiltCFG, Result.getStatus());

  CFG *cfg = Result.getCFG();

  // Basic correctness checks.
  ASSERT_EQ(cfg->size(), 6u);

  auto &EntryBlock = cfg->getEntry();

  CFGInterval I = buildInterval(*cfg, EntryBlock);
  EXPECT_EQ(I.Blocks.size(), 6u);
}

using ::testing::UnorderedElementsAre;

TEST(BuildInterval, PartitionWhileMultipleIntervals) {

  const char *Code = R"(void f() {
                          int x = 3;
                          while (x >= 3)
                            --x;
                          x = x + x;
                        })";
  BuildResult Result = BuildCFG(Code);
  ASSERT_EQ(BuildResult::BuiltCFG, Result.getStatus());

  CFG *cfg = Result.getCFG();
  ASSERT_EQ(cfg->size(), 7u);

  auto *EntryBlock = &cfg->getEntry();
  CFGBlock *InitXBlock = *EntryBlock->succ_begin();
  CFGBlock *LoopHeadBlock = *InitXBlock->succ_begin();

  CFGInterval I1 = buildInterval(*cfg, *EntryBlock);
  EXPECT_THAT(I1.Blocks, UnorderedElementsAre(EntryBlock, InitXBlock));

  CFGInterval I2 = buildInterval(*cfg, *LoopHeadBlock);
  EXPECT_EQ(I2.Blocks.size(), 5u);
}

TEST(PartitionIntoIntervals, PartitionIfThenOneInterval) {
  const char *Code = R"(void f() {
                          int x = 3;
                          if (x > 3)
                            x = 2;
                          else
                            x = 7;
                          x = x + x;
                        })";
  BuildResult Result = BuildCFG(Code);
  ASSERT_EQ(BuildResult::BuiltCFG, Result.getStatus());

  CFG *cfg = Result.getCFG();
  ASSERT_EQ(cfg->size(), 6u);

  auto Intervals = partitionIntoIntervals(*cfg);
  EXPECT_EQ(Intervals.size(), 1u);
}

TEST(PartitionIntoIntervals, PartitionWhileTwoIntervals) {
  const char *Code = R"(void f() {
                          int x = 3;
                          while (x >= 3)
                            --x;
                          x = x + x;
                        })";
  BuildResult Result = BuildCFG(Code);
  ASSERT_EQ(BuildResult::BuiltCFG, Result.getStatus());

  CFG *cfg = Result.getCFG();
  ASSERT_EQ(cfg->size(), 7u);

  auto Intervals = partitionIntoIntervals(*cfg);
  EXPECT_EQ(Intervals.size(), 2u);
}

TEST(PartitionIntoIntervals, PartitionNestedWhileThreeIntervals) {
  const char *Code = R"(void f() {
                          int x = 3;
                          while (x >= 3) {
                            --x;
                            int y = x;
                            while (y > 0) --y;
                          }
                          x = x + x;
                        })";
  BuildResult Result = BuildCFG(Code);
  ASSERT_EQ(BuildResult::BuiltCFG, Result.getStatus());

  CFG *cfg = Result.getCFG();
  auto Intervals = partitionIntoIntervals(*cfg);
  EXPECT_EQ(Intervals.size(), 3u);
}

TEST(PartitionIntoIntervals, PartitionSequentialWhileThreeIntervals) {
  const char *Code = R"(void f() {
                          int x = 3;
                          while (x >= 3) {
                            --x;
                          }
                          x = x + x;
                          int y = x;
                          while (y > 0) --y;
                        })";
  BuildResult Result = BuildCFG(Code);
  ASSERT_EQ(BuildResult::BuiltCFG, Result.getStatus());

  CFG *cfg = Result.getCFG();
  auto Intervals = partitionIntoIntervals(*cfg);
  EXPECT_EQ(Intervals.size(), 3u);
}

} // namespace
} // namespace analysis
} // namespace clang
