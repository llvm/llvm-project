//===- unittests/Tooling/RecursiveASTVisitorTests/CallbacksCompoundAssignOperator.cpp -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallbacksCommon.h"

TEST(RecursiveASTVisitor, StmtCallbacks_TraverseCompoundAssignOperator) {
  class RecordingVisitor : public RecordingVisitorBase<RecordingVisitor> {
  public:
    RecordingVisitor(ShouldTraversePostOrder ShouldTraversePostOrderValue)
        : RecordingVisitorBase(ShouldTraversePostOrderValue) {}

    bool TraverseCompoundAssignOperator(CompoundAssignOperator *CAO) {
      recordCallback(__func__, CAO, [&]() {
        RecordingVisitorBase::TraverseCompoundAssignOperator(CAO);
      });
      return true;
    }

    bool WalkUpFromStmt(Stmt *S) {
      recordCallback(__func__, S,
                     [&]() { RecordingVisitorBase::WalkUpFromStmt(S); });
      return true;
    }
  };

  StringRef Code = R"cpp(
void test(int a) {
  1;
  a += 2;
  3;
}
)cpp";

  EXPECT_TRUE(visitorCallbackLogEqual(
      RecordingVisitor(ShouldTraversePostOrder::No), Code,
      R"txt(
WalkUpFromStmt CompoundStmt
WalkUpFromStmt IntegerLiteral(1)
TraverseCompoundAssignOperator CompoundAssignOperator(+=)
  WalkUpFromStmt CompoundAssignOperator(+=)
  WalkUpFromStmt DeclRefExpr(a)
  WalkUpFromStmt IntegerLiteral(2)
WalkUpFromStmt IntegerLiteral(3)
)txt"));

  EXPECT_TRUE(visitorCallbackLogEqual(
      RecordingVisitor(ShouldTraversePostOrder::Yes), Code,
      R"txt(
WalkUpFromStmt IntegerLiteral(1)
TraverseCompoundAssignOperator CompoundAssignOperator(+=)
  WalkUpFromStmt DeclRefExpr(a)
  WalkUpFromStmt IntegerLiteral(2)
  WalkUpFromStmt CompoundAssignOperator(+=)
WalkUpFromStmt IntegerLiteral(3)
WalkUpFromStmt CompoundStmt
)txt"));
}

TEST(
    RecursiveASTVisitor,
    StmtCallbacks_TraverseCompoundAssignOperator_WalkUpFromCompoundAssignOperator) {
  class RecordingVisitor : public RecordingVisitorBase<RecordingVisitor> {
  public:
    RecordingVisitor(ShouldTraversePostOrder ShouldTraversePostOrderValue)
        : RecordingVisitorBase(ShouldTraversePostOrderValue) {}

    bool TraverseCompoundAssignOperator(CompoundAssignOperator *CAO) {
      recordCallback(__func__, CAO, [&]() {
        RecordingVisitorBase::TraverseCompoundAssignOperator(CAO);
      });
      return true;
    }

    bool WalkUpFromStmt(Stmt *S) {
      recordCallback(__func__, S,
                     [&]() { RecordingVisitorBase::WalkUpFromStmt(S); });
      return true;
    }

    bool WalkUpFromExpr(Expr *E) {
      recordCallback(__func__, E,
                     [&]() { RecordingVisitorBase::WalkUpFromExpr(E); });
      return true;
    }

    bool WalkUpFromCompoundAssignOperator(CompoundAssignOperator *CAO) {
      recordCallback(__func__, CAO, [&]() {
        RecordingVisitorBase::WalkUpFromCompoundAssignOperator(CAO);
      });
      return true;
    }
  };

  StringRef Code = R"cpp(
void test(int a) {
  1;
  a += 2;
  3;
}
)cpp";

  EXPECT_TRUE(visitorCallbackLogEqual(
      RecordingVisitor(ShouldTraversePostOrder::No), Code,
      R"txt(
WalkUpFromStmt CompoundStmt
WalkUpFromExpr IntegerLiteral(1)
  WalkUpFromStmt IntegerLiteral(1)
TraverseCompoundAssignOperator CompoundAssignOperator(+=)
  WalkUpFromCompoundAssignOperator CompoundAssignOperator(+=)
    WalkUpFromExpr CompoundAssignOperator(+=)
      WalkUpFromStmt CompoundAssignOperator(+=)
  WalkUpFromExpr DeclRefExpr(a)
    WalkUpFromStmt DeclRefExpr(a)
  WalkUpFromExpr IntegerLiteral(2)
    WalkUpFromStmt IntegerLiteral(2)
WalkUpFromExpr IntegerLiteral(3)
  WalkUpFromStmt IntegerLiteral(3)
)txt"));

  EXPECT_TRUE(visitorCallbackLogEqual(
      RecordingVisitor(ShouldTraversePostOrder::Yes), Code,
      R"txt(
WalkUpFromExpr IntegerLiteral(1)
  WalkUpFromStmt IntegerLiteral(1)
TraverseCompoundAssignOperator CompoundAssignOperator(+=)
  WalkUpFromExpr DeclRefExpr(a)
    WalkUpFromStmt DeclRefExpr(a)
  WalkUpFromExpr IntegerLiteral(2)
    WalkUpFromStmt IntegerLiteral(2)
  WalkUpFromCompoundAssignOperator CompoundAssignOperator(+=)
    WalkUpFromExpr CompoundAssignOperator(+=)
      WalkUpFromStmt CompoundAssignOperator(+=)
WalkUpFromExpr IntegerLiteral(3)
  WalkUpFromStmt IntegerLiteral(3)
WalkUpFromStmt CompoundStmt
)txt"));
}

TEST(RecursiveASTVisitor, StmtCallbacks_WalkUpFromCompoundAssignOperator) {
  class RecordingVisitor : public RecordingVisitorBase<RecordingVisitor> {
  public:
    RecordingVisitor(ShouldTraversePostOrder ShouldTraversePostOrderValue)
        : RecordingVisitorBase(ShouldTraversePostOrderValue) {}

    bool WalkUpFromStmt(Stmt *S) {
      recordCallback(__func__, S,
                     [&]() { RecordingVisitorBase::WalkUpFromStmt(S); });
      return true;
    }

    bool WalkUpFromExpr(Expr *E) {
      recordCallback(__func__, E,
                     [&]() { RecordingVisitorBase::WalkUpFromExpr(E); });
      return true;
    }

    bool WalkUpFromCompoundAssignOperator(CompoundAssignOperator *CAO) {
      recordCallback(__func__, CAO, [&]() {
        RecordingVisitorBase::WalkUpFromCompoundAssignOperator(CAO);
      });
      return true;
    }
  };

  StringRef Code = R"cpp(
void test(int a) {
  1;
  a += 2;
  3;
}
)cpp";

  EXPECT_TRUE(visitorCallbackLogEqual(
      RecordingVisitor(ShouldTraversePostOrder::No), Code,
      R"txt(
WalkUpFromStmt CompoundStmt
WalkUpFromExpr IntegerLiteral(1)
  WalkUpFromStmt IntegerLiteral(1)
WalkUpFromCompoundAssignOperator CompoundAssignOperator(+=)
  WalkUpFromExpr CompoundAssignOperator(+=)
    WalkUpFromStmt CompoundAssignOperator(+=)
WalkUpFromExpr DeclRefExpr(a)
  WalkUpFromStmt DeclRefExpr(a)
WalkUpFromExpr IntegerLiteral(2)
  WalkUpFromStmt IntegerLiteral(2)
WalkUpFromExpr IntegerLiteral(3)
  WalkUpFromStmt IntegerLiteral(3)
)txt"));

  EXPECT_TRUE(visitorCallbackLogEqual(
      RecordingVisitor(ShouldTraversePostOrder::Yes), Code,
      R"txt(
WalkUpFromExpr IntegerLiteral(1)
  WalkUpFromStmt IntegerLiteral(1)
WalkUpFromExpr DeclRefExpr(a)
  WalkUpFromStmt DeclRefExpr(a)
WalkUpFromExpr IntegerLiteral(2)
  WalkUpFromStmt IntegerLiteral(2)
WalkUpFromCompoundAssignOperator CompoundAssignOperator(+=)
  WalkUpFromExpr CompoundAssignOperator(+=)
    WalkUpFromStmt CompoundAssignOperator(+=)
WalkUpFromExpr IntegerLiteral(3)
  WalkUpFromStmt IntegerLiteral(3)
WalkUpFromStmt CompoundStmt
)txt"));
}
