//===- unittests/Analysis/IntervalPartitionTest.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/IntervalPartition.h"
#include "CFGBuildResult.h"
#include "clang/Analysis/CFG.h"
#include "llvm/Support/raw_ostream.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <type_traits>
#include <variant>

namespace clang {

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const std::vector<const CFGBlock *> &Nodes) {
  OS << "Blocks{";
  for (const auto *B : Nodes)
    OS << B->getBlockID() << ", ";
  OS << "}";
  return OS;
}

void PrintTo(const std::vector<const CFGBlock *> &Nodes, std::ostream *OS) {
  std::string Result;
  llvm::raw_string_ostream StringOS(Result);
  StringOS << Nodes;
  *OS << Result;
}

namespace internal {
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const CFGIntervalNode &I) {
  OS << "Interval{ID = " << I.ID << ", ";
  OS << "Blocks{";
  for (const auto *B : I.Nodes)
    OS << B->getBlockID() << ", ";
  OS << "}, Pre{";
  for (const auto *P : I.Predecessors)
    OS << P->ID << ",";
  OS << "}, Succ{";
  for (const auto *P : I.Successors)
    OS << P->ID << ",";
  OS << "}}";
  return OS;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const CFGIntervalGraph &G) {
  OS << "Intervals{";
  for (const auto &I : G) {
    OS << I << ", ";
  }
  OS << "}";
  return OS;
}

void PrintTo(const CFGIntervalNode &I, std::ostream *OS) {
  std::string Result;
  llvm::raw_string_ostream StringOS(Result);
  StringOS << I;
  *OS << Result;
}

void PrintTo(const CFGIntervalGraph &G, std::ostream *OS) {
  *OS << "Intervals{";
  for (const auto &I : G) {
    PrintTo(I, OS);
    *OS << ", ";
  }
  *OS << "}";
}
} // namespace internal

namespace {

using ::clang::analysis::BuildCFG;
using ::clang::analysis::BuildResult;
using ::clang::internal::buildInterval;
using ::clang::internal::partitionIntoIntervals;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Optional;
using ::testing::Property;
using ::testing::UnorderedElementsAre;

MATCHER_P(intervalID, ID, "") { return arg->ID == ID; }

template <typename... T> auto blockIDs(T... IDs) {
  return UnorderedElementsAre(Property(&CFGBlock::getBlockID, IDs)...);
}

template <typename... T> auto blockOrder(T... IDs) {
  return ElementsAre(Property(&CFGBlock::getBlockID, IDs)...);
}

MATCHER_P3(isInterval, ID, Preds, Succs, "") {
  return testing::Matches(ID)(arg.ID) &&
         testing::Matches(Preds)(arg.Predecessors) &&
         testing::Matches(Succs)(arg.Successors);
}

MATCHER_P4(isInterval, ID, Nodes, Preds, Succs, "") {
  return testing::Matches(ID)(arg.ID) && testing::Matches(Nodes)(arg.Nodes) &&
         testing::Matches(Preds)(arg.Predecessors) &&
         testing::Matches(Succs)(arg.Successors);
}

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

  std::vector<const CFGBlock *> I = buildInterval(&EntryBlock);
  EXPECT_EQ(I.size(), 3u);
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

  std::vector<const CFGBlock *> I = buildInterval(&EntryBlock);
  EXPECT_EQ(I.size(), 6u);
}

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

  std::vector<const CFGBlock *> I1 = buildInterval(EntryBlock);
  EXPECT_THAT(I1, ElementsAre(EntryBlock, InitXBlock));

  std::vector<const CFGBlock *> I2 = buildInterval(LoopHeadBlock);
  EXPECT_EQ(I2.size(), 5u);
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

  auto Graph = partitionIntoIntervals(*Result.getCFG());
  EXPECT_EQ(Graph.size(), 1u);
  EXPECT_THAT(Graph, ElementsAre(isInterval(0, IsEmpty(), IsEmpty())));
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

  auto Graph = partitionIntoIntervals(*Result.getCFG());
  EXPECT_THAT(
      Graph,
      ElementsAre(
          isInterval(0, IsEmpty(), UnorderedElementsAre(intervalID(1u))),
          isInterval(1, UnorderedElementsAre(intervalID(0u)), IsEmpty())));
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

  auto Graph = partitionIntoIntervals(*Result.getCFG());
  EXPECT_THAT(
      Graph,
      ElementsAre(
          isInterval(0, IsEmpty(), UnorderedElementsAre(intervalID(1u))),
          isInterval(1, UnorderedElementsAre(intervalID(0u), intervalID(2u)),
                     UnorderedElementsAre(intervalID(2u))),
          isInterval(2, UnorderedElementsAre(intervalID(1u)),
                     UnorderedElementsAre(intervalID(1u)))));
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

  auto Graph = partitionIntoIntervals(*Result.getCFG());
  EXPECT_THAT(
      Graph,
      ElementsAre(
          isInterval(0, IsEmpty(), UnorderedElementsAre(intervalID(1u))),
          isInterval(1, UnorderedElementsAre(intervalID(0u)),
                     UnorderedElementsAre(intervalID(2u))),
          isInterval(2, UnorderedElementsAre(intervalID(1u)), IsEmpty())));
}

TEST(PartitionIntoIntervals, LimitReducibleSequentialWhile) {
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

  auto Graph = partitionIntoIntervals(*Result.getCFG());
  ASSERT_THAT(
      Graph,
      ElementsAre(isInterval(0, blockOrder(9, 8), IsEmpty(),
                             UnorderedElementsAre(intervalID(1u))),
                  isInterval(1, blockOrder(7, 6, 4, 5),
                             UnorderedElementsAre(intervalID(0u)),
                             UnorderedElementsAre(intervalID(2u))),
                  isInterval(2, blockOrder(3, 2, 0, 1),
                             UnorderedElementsAre(intervalID(1u)), IsEmpty())));

  auto Graph2 = partitionIntoIntervals(Graph);
  EXPECT_THAT(Graph2, ElementsAre(isInterval(
                          0, blockOrder(9, 8, 7, 6, 4, 5, 3, 2, 0, 1),
                          IsEmpty(), IsEmpty())));
}

TEST(PartitionIntoIntervals, LimitReducibleNestedWhile) {
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

  auto Graph = partitionIntoIntervals(*Result.getCFG());
  ASSERT_THAT(Graph,
              ElementsAre(isInterval(0, blockOrder(9, 8), IsEmpty(),
                                     UnorderedElementsAre(intervalID(1u))),
                          isInterval(1, blockOrder(7, 6, 1, 0),
                                     UnorderedElementsAre(intervalID(0u),
                                                          intervalID(2u)),
                                     UnorderedElementsAre(intervalID(2u))),
                          isInterval(2, blockOrder(5, 4, 2, 3),
                                     UnorderedElementsAre(intervalID(1u)),
                                     UnorderedElementsAre(intervalID(1u)))));

  auto Graph2 = partitionIntoIntervals(Graph);
  EXPECT_THAT(
      Graph2,
      ElementsAre(isInterval(0, blockOrder(9, 8), IsEmpty(),
                             UnorderedElementsAre(intervalID(1u))),
                  isInterval(1, blockOrder(7, 6, 1, 0, 5, 4, 2, 3),
                             UnorderedElementsAre(intervalID(0u)), IsEmpty())));

  auto Graph3 = partitionIntoIntervals(Graph2);
  EXPECT_THAT(Graph3, ElementsAre(isInterval(
                          0, blockOrder(9, 8, 7, 6, 1, 0, 5, 4, 2, 3),
                          IsEmpty(), IsEmpty())));
}

TEST(GetIntervalWTO, SequentialWhile) {
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
  EXPECT_THAT(getIntervalWTO(*Result.getCFG()),
              Optional(blockOrder(9, 8, 7, 6, 4, 5, 3, 2, 0, 1)));
}

TEST(GetIntervalWTO, NestedWhile) {
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
  EXPECT_THAT(getIntervalWTO(*Result.getCFG()),
              Optional(blockOrder(9, 8, 7, 6, 1, 0, 5, 4, 2, 3)));
}

TEST(GetIntervalWTO, UnreachablePred) {
  const char *Code = R"(
  void target(bool Foo) {
    bool Bar = false;
    if (Foo)
      Bar = Foo;
    else
      __builtin_unreachable();
    (void)0;
  })";
  BuildResult Result = BuildCFG(Code);
  ASSERT_EQ(BuildResult::BuiltCFG, Result.getStatus());
  EXPECT_THAT(getIntervalWTO(*Result.getCFG()),
              Optional(blockOrder(5, 4, 3, 2, 1, 0)));
}

TEST(WTOCompare, UnreachableBlock) {
  const char *Code = R"(
    void target() {
      while (true) {}
      (void)0;
      /*[[p]]*/
    })";
  BuildResult Result = BuildCFG(Code);
  ASSERT_EQ(BuildResult::BuiltCFG, Result.getStatus());
  std::optional<WeakTopologicalOrdering> WTO = getIntervalWTO(*Result.getCFG());
  ASSERT_THAT(WTO, Optional(blockOrder(4, 3, 2)));
  auto Cmp = WTOCompare(*WTO);
  const CFGBlock &Entry = Result.getCFG()->getEntry();
  const CFGBlock &Exit = Result.getCFG()->getExit();
  EXPECT_TRUE(Cmp(&Entry, &Exit));
}

} // namespace
} // namespace clang
