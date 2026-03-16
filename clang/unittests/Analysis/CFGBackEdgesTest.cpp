//===- unittests/Analysis/CFGBackEdgesTest.cpp - CFG backedges tests ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/CFGBackEdges.h"
#include "CFGBuildResult.h"
#include "clang/AST/Stmt.h"
#include "clang/Analysis/CFG.h"
#include "clang/Basic/LLVM.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace analysis {
namespace {

using ::testing::IsNull;
using ::testing::NotNull;
using ::testing::SizeIs;

TEST(CFGBackEdgesTest, NoBackedgesLinear) {
  const char *Code = R"cc(
    int f(int x) {
      l1:
        x++;
      l2:
        x++;
      return x;
    })cc";
  BuildResult Result = BuildCFG(Code);
  EXPECT_EQ(BuildResult::BuiltCFG, Result.getStatus());
  CFG *Cfg = Result.getCFG();
  ASSERT_THAT(Cfg, NotNull());

  auto BackEdges = findCFGBackEdges(*Cfg);
  EXPECT_TRUE(BackEdges.empty());
}

TEST(CFGBackEdgesTest, NoBackedgesOnlyCrossEdge) {
  const char *Code = R"cc(
    int f(int x) {
      if (x > 0)
        x++;
      else
        x--;
      return x;
    })cc";
  BuildResult Result = BuildCFG(Code);
  EXPECT_EQ(BuildResult::BuiltCFG, Result.getStatus());
  CFG *Cfg = Result.getCFG();
  ASSERT_THAT(Cfg, NotNull());

  auto BackEdges = findCFGBackEdges(*Cfg);
  EXPECT_TRUE(BackEdges.empty());
}

TEST(CFGBackEdgesTest, NoBackedgesWithUnreachableSuccessorForSwitch) {
  const char *Code = R"cc(
    enum class Kind { A, B };

    void f(Kind kind) {
      switch(kind) {
      case Kind::A: return 0;
      case Kind::B: break;
      }
      return 1;
  })cc";
  BuildResult Result = BuildCFG(Code);
  EXPECT_EQ(BuildResult::BuiltCFG, Result.getStatus());
  CFG *Cfg = Result.getCFG();
  ASSERT_THAT(Cfg, NotNull());

  auto BackEdges = findCFGBackEdges(*Cfg);
  EXPECT_TRUE(BackEdges.empty());
}

TEST(CFGBackEdgesTest, ForLoop) {
  const char *Code = R"cc(
    void f(int n) {
      for (int i = 0; i < n; ++i) {}
    })cc";
  BuildResult Result = BuildCFG(Code);
  EXPECT_EQ(BuildResult::BuiltCFG, Result.getStatus());
  CFG *Cfg = Result.getCFG();
  ASSERT_THAT(Cfg, NotNull());

  // Finds one backedge, which is the one looping back to the loop header
  // (has a loop target).
  auto BackEdges = findCFGBackEdges(*Cfg);
  EXPECT_THAT(BackEdges, SizeIs(1));
  EXPECT_THAT(BackEdges.begin()->first->getLoopTarget(), NotNull());
}

TEST(CFGBackEdgesTest, WhileLoop) {
  const char *Code = R"cc(
    void f(int n) {
      int i = 0;
      while (i < n) { ++i; }
    })cc";
  BuildResult Result = BuildCFG(Code);
  EXPECT_EQ(BuildResult::BuiltCFG, Result.getStatus());
  CFG *Cfg = Result.getCFG();
  ASSERT_THAT(Cfg, NotNull());

  auto BackEdges = findCFGBackEdges(*Cfg);
  EXPECT_THAT(BackEdges, SizeIs(1));
  EXPECT_THAT(BackEdges.begin()->first->getLoopTarget(), NotNull());
}

TEST(CFGBackEdgesTest, DoWhileLoop) {
  const char *Code = R"cc(
    void f(int n) {
      int i = 0;
      do { ++i; } while (i < n);
    })cc";
  BuildResult Result = BuildCFG(Code);
  EXPECT_EQ(BuildResult::BuiltCFG, Result.getStatus());
  CFG *Cfg = Result.getCFG();
  ASSERT_THAT(Cfg, NotNull());

  auto BackEdges = findCFGBackEdges(*Cfg);
  EXPECT_THAT(BackEdges, SizeIs(1));
  EXPECT_THAT(BackEdges.begin()->first->getLoopTarget(), NotNull());
}

TEST(CFGBackEdgesTest, GotoLoop) {
  const char *Code = R"cc(
    void f(int n) {
      int i = 0;
    loop:
      if (i < n) {
        ++i;
        goto loop;
      }
    })cc";
  BuildResult Result = BuildCFG(Code);
  EXPECT_EQ(BuildResult::BuiltCFG, Result.getStatus());
  CFG *Cfg = Result.getCFG();
  ASSERT_THAT(Cfg, NotNull());

  // Finds one backedge, but since it's an unstructured loop, the loop target is
  // null. Instead, the node has a goto terminator.
  auto BackEdges = findCFGBackEdges(*Cfg);
  EXPECT_THAT(BackEdges, SizeIs(1));
  EXPECT_THAT(BackEdges.begin()->first->getLoopTarget(), IsNull());
  EXPECT_TRUE(isa<GotoStmt>(BackEdges.begin()->first->getTerminatorStmt()));
}

TEST(CFGBackEdgesTest, WhileWithContinueLoop) {
  const char *Code = R"cc(
    void f(int n) {
      int i = 0;
      while (i < n) {
        ++i;
        if (i == 5) continue;
        if (i == 10) break;
        i *= 2;
      }
    })cc";
  BuildResult Result = BuildCFG(Code);
  EXPECT_EQ(BuildResult::BuiltCFG, Result.getStatus());
  CFG *Cfg = Result.getCFG();
  ASSERT_THAT(Cfg, NotNull());

  auto BackEdges = findCFGBackEdges(*Cfg);
  EXPECT_THAT(BackEdges, SizeIs(testing::Gt(0)));
  for (const auto &[From, To] : BackEdges)
    EXPECT_THAT(From->getLoopTarget(), NotNull());
}

TEST(CFGBackEdgesTest, NestedForLoop) {
  const char *Code = R"cc(
    void f(int n) {
      for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {}
      }
    })cc";
  BuildResult Result = BuildCFG(Code);
  EXPECT_EQ(BuildResult::BuiltCFG, Result.getStatus());
  CFG *Cfg = Result.getCFG();
  ASSERT_THAT(Cfg, NotNull());

  auto BackEdges = findCFGBackEdges(*Cfg);
  EXPECT_THAT(BackEdges, SizeIs(2));
  auto It = BackEdges.begin();
  auto *FirstLoopTarget = It->first->getLoopTarget();
  EXPECT_THAT(FirstLoopTarget, NotNull());
  ++It;
  auto *SecondLoopTarget = It->first->getLoopTarget();
  EXPECT_THAT(SecondLoopTarget, NotNull());
  EXPECT_NE(FirstLoopTarget, SecondLoopTarget);
}

TEST(CFGBackEdgesTest, IrreducibleCFG) {
  const char *Code = R"cc(
    void f(int cond) {
      if (cond) goto L1;
    L0:
      goto L1;
    L1:
      goto L0;
    })cc";
  BuildResult Result = BuildCFG(Code);
  EXPECT_EQ(BuildResult::BuiltCFG, Result.getStatus());
  CFG *Cfg = Result.getCFG();
  ASSERT_THAT(Cfg, NotNull());

  auto BackEdges = findCFGBackEdges(*Cfg);
  // In an irreducible CFG, we still expect to find a back edge.
  EXPECT_THAT(BackEdges, SizeIs(1));
  EXPECT_TRUE(isa<GotoStmt>(BackEdges.begin()->first->getTerminatorStmt()));
}

TEST(CFGBackEdgesTest, FirstBackedgeIsNotGoto) {
  const char *Code = R"cc(
    void f(int x, int y) {
      if (x > y) {
      } else {
      L1:
        --x;
        if (x == 0) return;
      }
      goto L1;
    })cc";
  BuildResult Result = BuildCFG(Code);
  EXPECT_EQ(BuildResult::BuiltCFG, Result.getStatus());
  CFG *Cfg = Result.getCFG();

  auto BackEdges = findCFGBackEdges(*Cfg);
  EXPECT_THAT(BackEdges, SizeIs(1));
  // We might find a backedge where the source block doesn't terminate with
  // a `goto`, due to the DFS search order. For example:
  //
  // B_entry: `if (x > y)`
  //   \--then--> B1: `<empty>`
  //      --> B2: `goto L1`
  //        --> B3: `--x; if (x == 0)`
  //          \--then--> B4 `return` --> B_exit.
  //          \--else--> B2: ... (the `if`'s else is a backedge from B3 to B2!)
  //   \--else--> B3: ...
  EXPECT_FALSE(isa<GotoStmt>(BackEdges.begin()->first->getTerminatorStmt()));
}

TEST(CFGBackEdgesTest, FindNonStructuredLoopBackedgeNodes) {
  const char *Code = R"cc(
    void f(int n) {
      for (int i = 0; i < n; ++i) {
        int j = 0;
        inner_loop:
        if (j < n) {
          ++j;
          goto inner_loop;
        }
      }
    })cc";
  BuildResult Result = BuildCFG(Code);
  EXPECT_EQ(BuildResult::BuiltCFG, Result.getStatus());
  CFG *Cfg = Result.getCFG();
  ASSERT_THAT(Cfg, NotNull());

  // Finds just the goto backedge, and not the for-loop backedge.
  auto BackEdgeNodes = findNonStructuredLoopBackedgeNodes(*Cfg);
  EXPECT_THAT(BackEdgeNodes, SizeIs(1));
  const CFGBlock *Node = *BackEdgeNodes.begin();
  EXPECT_EQ(Node->getLoopTarget(), nullptr);
  EXPECT_TRUE(isa<GotoStmt>(Node->getTerminatorStmt()));
}

TEST(CFGBackEdgesTest, IsBackedgeCFGNode) {
  const char *Code = R"cc(
    void f(int n) {
      for (int i = 0; i < n; ++i) {
        int j = 0;
        inner_loop:
        if (j < n) {
          ++j;
          goto inner_loop;
        }
      }
    })cc";
  BuildResult Result = BuildCFG(Code);
  EXPECT_EQ(BuildResult::BuiltCFG, Result.getStatus());
  CFG *Cfg = Result.getCFG();
  ASSERT_THAT(Cfg, NotNull());

  auto BackEdgeNodes = findNonStructuredLoopBackedgeNodes(*Cfg);

  // `isBackedgeCFGNode` should be true for both the for-loop backedge node and
  // goto backedge nodes.
  const CFGBlock *ForLoopBackedgeNode = nullptr;
  const CFGBlock *GotoBackedgeNode = nullptr;
  for (const CFGBlock *Block : *Cfg) {
    if (Block->getLoopTarget() != nullptr) {
      ForLoopBackedgeNode = Block;
    } else if (Block->getTerminatorStmt() != nullptr &&
               isa<GotoStmt>(Block->getTerminatorStmt())) {
      GotoBackedgeNode = Block;
    }
  }
  ASSERT_THAT(ForLoopBackedgeNode, NotNull());
  ASSERT_THAT(GotoBackedgeNode, NotNull());
  EXPECT_TRUE(isBackedgeCFGNode(*ForLoopBackedgeNode, BackEdgeNodes));
  EXPECT_TRUE(isBackedgeCFGNode(*GotoBackedgeNode, BackEdgeNodes));
}

} // namespace
} // namespace analysis
} // namespace clang
