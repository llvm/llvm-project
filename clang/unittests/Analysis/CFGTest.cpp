//===- unittests/Analysis/CFGTest.cpp - CFG tests -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/CFG.h"
#include "CFGBuildResult.h"
#include "clang/AST/Decl.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Analysis/Analyses/IntervalPartition.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Analysis/FlowSensitive/DataflowWorklist.h"
#include "clang/Tooling/Tooling.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <string>
#include <vector>

namespace clang {
namespace analysis {
namespace {

// Constructing a CFG for a range-based for over a dependent type fails (but
// should not crash).
TEST(CFG, RangeBasedForOverDependentType) {
  const char *Code = "class Foo;\n"
                     "template <typename T>\n"
                     "void f(const T &Range) {\n"
                     "  for (const Foo *TheFoo : Range) {\n"
                     "  }\n"
                     "}\n";
  EXPECT_EQ(BuildResult::SawFunctionBody, BuildCFG(Code).getStatus());
}

TEST(CFG, StaticInitializerLastCondition) {
  const char *Code = "void f() {\n"
                     "  int i = 5 ;\n"
                     "  static int j = 3 ;\n"
                     "}\n";
  CFG::BuildOptions Options;
  Options.AddStaticInitBranches = true;
  Options.setAllAlwaysAdd();
  BuildResult B = BuildCFG(Code, Options);
  EXPECT_EQ(BuildResult::BuiltCFG, B.getStatus());
  EXPECT_EQ(1u, B.getCFG()->getEntry().succ_size());
  CFGBlock *Block = *B.getCFG()->getEntry().succ_begin();
  EXPECT_TRUE(isa<DeclStmt>(Block->getTerminatorStmt()));
  EXPECT_EQ(nullptr, Block->getLastCondition());
}

// Constructing a CFG containing a delete expression on a dependent type should
// not crash.
TEST(CFG, DeleteExpressionOnDependentType) {
  const char *Code = "template<class T>\n"
                     "void f(T t) {\n"
                     "  delete t;\n"
                     "}\n";
  EXPECT_EQ(BuildResult::BuiltCFG, BuildCFG(Code).getStatus());
}

// Constructing a CFG on a function template with a variable of incomplete type
// should not crash.
TEST(CFG, VariableOfIncompleteType) {
  const char *Code = "template<class T> void f() {\n"
                     "  class Undefined;\n"
                     "  Undefined u;\n"
                     "}\n";
  EXPECT_EQ(BuildResult::BuiltCFG, BuildCFG(Code).getStatus());
}

// Constructing a CFG with a dependent base should not crash.
TEST(CFG, DependantBaseAddImplicitDtors) {
  const char *Code = R"(
    template <class T>
    struct Base {
      virtual ~Base() {}
    };

    template <typename T>
    struct Derived : public Base<T> {
      virtual ~Derived() {}
    };
  )";
  CFG::BuildOptions Options;
  Options.AddImplicitDtors = true;
  Options.setAllAlwaysAdd();
  EXPECT_EQ(BuildResult::BuiltCFG,
            BuildCFG(Code, Options, ast_matchers::hasName("~Derived<T>"))
                .getStatus());
}

TEST(CFG, SwitchCoveredEnumNoDefault) {
  const char *Code = R"(
    enum class E {E1, E2};
    int f(E e) {
      switch(e) {
        case E::E1:
          return 1;
        case E::E2:
          return 2;
      }
      return 0;
    }
  )";
  CFG::BuildOptions Options;
  Options.AssumeReachableDefaultInSwitchStatements = true;
  BuildResult B = BuildCFG(Code, Options);
  ASSERT_EQ(BuildResult::BuiltCFG, B.getStatus());

  // [B5 (ENTRY)]
  //   Succs (1): B2
  //
  // [B1]
  //   1: 0
  //   2: return [B1.1];
  //   Preds (1): B2
  //   Succs (1): B0
  //
  // [B2]
  //   1: e (ImplicitCastExpr, LValueToRValue, E)
  //   T: switch [B2.1]
  //   Preds (1): B5
  //   Succs (3): B3 B4 B1
  //
  // [B3]
  //  case E::E2:
  //   1: 2
  //   2: return [B3.1];
  //   Preds (1): B2
  //   Succs (1): B0
  //
  // [B4]
  //  case E::E1:
  //   1: 1
  //   2: return [B4.1];
  //   Preds (1): B2
  //   Succs (1): B0
  //
  // [B0 (EXIT)]
  //   Preds (3): B1 B3 B4

  auto *CFG = B.getCFG();
  const auto &Entry = CFG->getEntry();
  ASSERT_EQ(1u, Entry.succ_size());
  // First successor of Entry is the switch
  CFGBlock *SwitchBlock = *Entry.succ_begin();
  ASSERT_EQ(3u, SwitchBlock->succ_size());
  // Last successor of the switch is after the switch
  auto NoCaseSucc = SwitchBlock->succ_rbegin();
  EXPECT_TRUE(NoCaseSucc->isReachable());

  // Checking that the same node is Unreachable without this setting
  Options.AssumeReachableDefaultInSwitchStatements = false;
  B = BuildCFG(Code, Options);
  ASSERT_EQ(BuildResult::BuiltCFG, B.getStatus());

  const auto &Entry2 = B.getCFG()->getEntry();
  ASSERT_EQ(1u, Entry2.succ_size());
  CFGBlock *SwitchBlock2 = *Entry2.succ_begin();
  ASSERT_EQ(3u, SwitchBlock2->succ_size());
  auto NoCaseSucc2 = SwitchBlock2->succ_rbegin();
  EXPECT_FALSE(NoCaseSucc2->isReachable());
}

TEST(CFG, SwitchCoveredEnumWithDefault) {
  const char *Code = R"(
    enum class E {E1, E2};
    int f(E e) {
      switch(e) {
        case E::E1:
          return 1;
        case E::E2:
          return 2;
        default:
          return 0;
      }
      return -1;
    }
  )";
  CFG::BuildOptions Options;
  Options.AssumeReachableDefaultInSwitchStatements = true;
  BuildResult B = BuildCFG(Code, Options);
  ASSERT_EQ(BuildResult::BuiltCFG, B.getStatus());

  // [B6 (ENTRY)]
  //   Succs (1): B2
  //
  // [B1]
  //   1: -1
  //   2: return [B1.1];
  //   Succs (1): B0
  //
  // [B2]
  //   1: e (ImplicitCastExpr, LValueToRValue, E)
  //   T: switch [B2.1]
  //   Preds (1): B6
  //   Succs (3): B4 B5 B3
  //
  // [B3]
  //  default:
  //   1: 0
  //   2: return [B3.1];
  //   Preds (1): B2
  //   Succs (1): B0
  //
  // [B4]
  //  case E::E2:
  //   1: 2
  //   2: return [B4.1];
  //   Preds (1): B2
  //   Succs (1): B0
  //
  // [B5]
  //  case E::E1:
  //   1: 1
  //   2: return [B5.1];
  //   Preds (1): B2
  //   Succs (1): B0
  //
  // [B0 (EXIT)]
  //   Preds (4): B1 B3 B4 B5

  const auto &Entry = B.getCFG()->getEntry();
  ASSERT_EQ(1u, Entry.succ_size());
  // First successor of Entry is the switch
  CFGBlock *SwitchBlock = *Entry.succ_begin();
  ASSERT_EQ(3u, SwitchBlock->succ_size());
  // Last successor of the switch is the default branch
  auto defaultBlock = SwitchBlock->succ_rbegin();
  EXPECT_TRUE(defaultBlock->isReachable());

  // Checking that the same node is Unreachable without this setting
  Options.AssumeReachableDefaultInSwitchStatements = false;
  B = BuildCFG(Code, Options);
  ASSERT_EQ(BuildResult::BuiltCFG, B.getStatus());

  const auto &Entry2 = B.getCFG()->getEntry();
  ASSERT_EQ(1u, Entry2.succ_size());
  CFGBlock *SwitchBlock2 = *Entry2.succ_begin();
  ASSERT_EQ(3u, SwitchBlock2->succ_size());
  auto defaultBlock2 = SwitchBlock2->succ_rbegin();
  EXPECT_FALSE(defaultBlock2->isReachable());
}

TEST(CFG, IsLinear) {
  auto expectLinear = [](bool IsLinear, const char *Code) {
    BuildResult B = BuildCFG(Code);
    EXPECT_EQ(BuildResult::BuiltCFG, B.getStatus());
    EXPECT_EQ(IsLinear, B.getCFG()->isLinear());
  };

  expectLinear(true, "void foo() {}");
  expectLinear(true, "void foo() { if (true) return; }");
  expectLinear(true, "void foo() { if constexpr (false); }");
  expectLinear(false, "void foo(bool coin) { if (coin) return; }");
  expectLinear(false, "void foo() { for(;;); }");
  expectLinear(false, "void foo() { do {} while (true); }");
  expectLinear(true, "void foo() { do {} while (false); }");
  expectLinear(true, "void foo() { foo(); }"); // Recursion is not our problem.
}

TEST(CFG, ElementRefIterator) {
  const char *Code = R"(void f() {
                          int i;
                          int j;
                          i = 5;
                          i = 6;
                          j = 7;
                        })";

  BuildResult B = BuildCFG(Code);
  EXPECT_EQ(BuildResult::BuiltCFG, B.getStatus());
  CFG *Cfg = B.getCFG();

  // [B2 (ENTRY)]
  //   Succs (1): B1

  // [B1]
  //   1: int i;
  //   2: int j;
  //   3: i = 5
  //   4: i = 6
  //   5: j = 7
  //   Preds (1): B2
  //   Succs (1): B0

  // [B0 (EXIT)]
  //   Preds (1): B1
  CFGBlock *MainBlock = *(Cfg->begin() + 1);

  constexpr CFGBlock::ref_iterator::difference_type MainBlockSize = 4;

  // The rest of this test looks totally insane, but there iterators are
  // templates under the hood, to it's important to instantiate everything for
  // proper converage.

  // Non-reverse, non-const version
  size_t Index = 0;
  for (CFGBlock::CFGElementRef ElementRef : MainBlock->refs()) {
    EXPECT_EQ(ElementRef.getParent(), MainBlock);
    EXPECT_EQ(ElementRef.getIndexInBlock(), Index);
    EXPECT_TRUE(ElementRef->getAs<CFGStmt>());
    EXPECT_TRUE((*ElementRef).getAs<CFGStmt>());
    EXPECT_EQ(ElementRef.getParent(), MainBlock);
    ++Index;
  }
  EXPECT_TRUE(*MainBlock->ref_begin() < *(MainBlock->ref_begin() + 1));
  EXPECT_TRUE(*MainBlock->ref_begin() == *MainBlock->ref_begin());
  EXPECT_FALSE(*MainBlock->ref_begin() != *MainBlock->ref_begin());

  EXPECT_TRUE(MainBlock->ref_begin() < MainBlock->ref_begin() + 1);
  EXPECT_TRUE(MainBlock->ref_begin() == MainBlock->ref_begin());
  EXPECT_FALSE(MainBlock->ref_begin() != MainBlock->ref_begin());
  EXPECT_EQ(MainBlock->ref_end() - MainBlock->ref_begin(), MainBlockSize + 1);
  EXPECT_EQ(MainBlock->ref_end() - MainBlockSize - 1, MainBlock->ref_begin());
  EXPECT_EQ(MainBlock->ref_begin() + MainBlockSize + 1, MainBlock->ref_end());
  EXPECT_EQ(MainBlock->ref_begin()++, MainBlock->ref_begin());
  EXPECT_EQ(++(MainBlock->ref_begin()), MainBlock->ref_begin() + 1);

  // Non-reverse, non-const version
  const CFGBlock *CMainBlock = MainBlock;
  Index = 0;
  for (CFGBlock::ConstCFGElementRef ElementRef : CMainBlock->refs()) {
    EXPECT_EQ(ElementRef.getParent(), CMainBlock);
    EXPECT_EQ(ElementRef.getIndexInBlock(), Index);
    EXPECT_TRUE(ElementRef->getAs<CFGStmt>());
    EXPECT_TRUE((*ElementRef).getAs<CFGStmt>());
    EXPECT_EQ(ElementRef.getParent(), MainBlock);
    ++Index;
  }
  EXPECT_TRUE(*CMainBlock->ref_begin() < *(CMainBlock->ref_begin() + 1));
  EXPECT_TRUE(*CMainBlock->ref_begin() == *CMainBlock->ref_begin());
  EXPECT_FALSE(*CMainBlock->ref_begin() != *CMainBlock->ref_begin());

  EXPECT_TRUE(CMainBlock->ref_begin() < CMainBlock->ref_begin() + 1);
  EXPECT_TRUE(CMainBlock->ref_begin() == CMainBlock->ref_begin());
  EXPECT_FALSE(CMainBlock->ref_begin() != CMainBlock->ref_begin());
  EXPECT_EQ(CMainBlock->ref_end() - CMainBlock->ref_begin(), MainBlockSize + 1);
  EXPECT_EQ(CMainBlock->ref_end() - MainBlockSize - 1, CMainBlock->ref_begin());
  EXPECT_EQ(CMainBlock->ref_begin() + MainBlockSize + 1, CMainBlock->ref_end());
  EXPECT_EQ(CMainBlock->ref_begin()++, CMainBlock->ref_begin());
  EXPECT_EQ(++(CMainBlock->ref_begin()), CMainBlock->ref_begin() + 1);

  // Reverse, non-const version
  Index = MainBlockSize;
  for (CFGBlock::CFGElementRef ElementRef : MainBlock->rrefs()) {
    EXPECT_EQ(ElementRef.getParent(), MainBlock);
    EXPECT_EQ(ElementRef.getIndexInBlock(), Index);
    EXPECT_TRUE(ElementRef->getAs<CFGStmt>());
    EXPECT_TRUE((*ElementRef).getAs<CFGStmt>());
    EXPECT_EQ(ElementRef.getParent(), MainBlock);
    --Index;
  }
  EXPECT_FALSE(*MainBlock->rref_begin() < *(MainBlock->rref_begin() + 1));
  EXPECT_TRUE(*MainBlock->rref_begin() == *MainBlock->rref_begin());
  EXPECT_FALSE(*MainBlock->rref_begin() != *MainBlock->rref_begin());

  EXPECT_TRUE(MainBlock->rref_begin() < MainBlock->rref_begin() + 1);
  EXPECT_TRUE(MainBlock->rref_begin() == MainBlock->rref_begin());
  EXPECT_FALSE(MainBlock->rref_begin() != MainBlock->rref_begin());
  EXPECT_EQ(MainBlock->rref_end() - MainBlock->rref_begin(), MainBlockSize + 1);
  EXPECT_EQ(MainBlock->rref_end() - MainBlockSize - 1, MainBlock->rref_begin());
  EXPECT_EQ(MainBlock->rref_begin() + MainBlockSize + 1, MainBlock->rref_end());
  EXPECT_EQ(MainBlock->rref_begin()++, MainBlock->rref_begin());
  EXPECT_EQ(++(MainBlock->rref_begin()), MainBlock->rref_begin() + 1);

  // Reverse, const version
  Index = MainBlockSize;
  for (CFGBlock::ConstCFGElementRef ElementRef : CMainBlock->rrefs()) {
    EXPECT_EQ(ElementRef.getParent(), CMainBlock);
    EXPECT_EQ(ElementRef.getIndexInBlock(), Index);
    EXPECT_TRUE(ElementRef->getAs<CFGStmt>());
    EXPECT_TRUE((*ElementRef).getAs<CFGStmt>());
    EXPECT_EQ(ElementRef.getParent(), MainBlock);
    --Index;
  }
  EXPECT_FALSE(*CMainBlock->rref_begin() < *(CMainBlock->rref_begin() + 1));
  EXPECT_TRUE(*CMainBlock->rref_begin() == *CMainBlock->rref_begin());
  EXPECT_FALSE(*CMainBlock->rref_begin() != *CMainBlock->rref_begin());

  EXPECT_TRUE(CMainBlock->rref_begin() < CMainBlock->rref_begin() + 1);
  EXPECT_TRUE(CMainBlock->rref_begin() == CMainBlock->rref_begin());
  EXPECT_FALSE(CMainBlock->rref_begin() != CMainBlock->rref_begin());
  EXPECT_EQ(CMainBlock->rref_end() - CMainBlock->rref_begin(),
            MainBlockSize + 1);
  EXPECT_EQ(CMainBlock->rref_end() - MainBlockSize - 1,
            CMainBlock->rref_begin());
  EXPECT_EQ(CMainBlock->rref_begin() + MainBlockSize + 1,
            CMainBlock->rref_end());
  EXPECT_EQ(CMainBlock->rref_begin()++, CMainBlock->rref_begin());
  EXPECT_EQ(++(CMainBlock->rref_begin()), CMainBlock->rref_begin() + 1);
}

TEST(CFG, Worklists) {
  const char *Code = "int f(bool cond) {\n"
                     "  int a = 5;\n"
                     "  while (a < 6)\n"
                     "    ++a;\n"
                     "  if (cond)\n"
                     "    a += 1;\n"
                     "  return a;\n"
                     "}\n";
  BuildResult B = BuildCFG(Code);
  EXPECT_EQ(BuildResult::BuiltCFG, B.getStatus());
  const FunctionDecl *Func = B.getFunc();
  AnalysisDeclContext AC(nullptr, Func);
  auto *CFG = AC.getCFG();

  std::vector<const CFGBlock *> ReferenceOrder;
  for (const auto *B : *AC.getAnalysis<PostOrderCFGView>())
    ReferenceOrder.push_back(B);

  {
    ForwardDataflowWorklist ForwardWorklist(*CFG, AC);
    for (const auto *B : *CFG)
      ForwardWorklist.enqueueBlock(B);

    std::vector<const CFGBlock *> ForwardNodes;
    while (const CFGBlock *B = ForwardWorklist.dequeue())
      ForwardNodes.push_back(B);

    EXPECT_EQ(ForwardNodes.size(), ReferenceOrder.size());
    EXPECT_TRUE(std::equal(ReferenceOrder.begin(), ReferenceOrder.end(),
                           ForwardNodes.begin()));
  }

  {
    using ::testing::ElementsAreArray;
    std::optional<WeakTopologicalOrdering> WTO = getIntervalWTO(*CFG);
    ASSERT_TRUE(WTO);
    WTOCompare WCmp(*WTO);
    WTODataflowWorklist WTOWorklist(*CFG, WCmp);
    for (const auto *B : *CFG)
      WTOWorklist.enqueueBlock(B);

    std::vector<const CFGBlock *> WTONodes;
    while (const CFGBlock *B = WTOWorklist.dequeue())
      WTONodes.push_back(B);

    EXPECT_THAT(WTONodes, ElementsAreArray(*WTO));
  }

  std::reverse(ReferenceOrder.begin(), ReferenceOrder.end());

  {
    BackwardDataflowWorklist BackwardWorklist(*CFG, AC);
    for (const auto *B : *CFG)
      BackwardWorklist.enqueueBlock(B);

    std::vector<const CFGBlock *> BackwardNodes;
    while (const CFGBlock *B = BackwardWorklist.dequeue())
      BackwardNodes.push_back(B);

    EXPECT_EQ(BackwardNodes.size(), ReferenceOrder.size());
    EXPECT_TRUE(std::equal(ReferenceOrder.begin(), ReferenceOrder.end(),
                           BackwardNodes.begin()));
  }
}

} // namespace
} // namespace analysis
} // namespace clang
