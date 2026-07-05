//===-- unittests/Tooling/RecursiveASTVisitorTests/CallbacksCallExpr.cpp --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallbacksCommon.h"

TEST(RecursiveASTVisitor, StmtCallbacks_TraverseCallExpr) {
  class RecordingVisitor : public RecordingVisitorBase<RecordingVisitor> {
  public:
    RecordingVisitor(ShouldTraversePostOrder ShouldTraversePostOrderValue)
        : RecordingVisitorBase(ShouldTraversePostOrderValue) {}

    bool TraverseCallExpr(CallExpr *CE) {
      recordCallback(__func__, CE,
                     [&]() { RecordingVisitorBase::TraverseCallExpr(CE); });
      return true;
    }

    bool WalkUpFromStmt(Stmt *S) {
      recordCallback(__func__, S,
                     [&]() { RecordingVisitorBase::WalkUpFromStmt(S); });
      return true;
    }
  };

  StringRef Code = R"cpp(
void add(int, int);
void test() {
  1;
  2 + 3;
  add(4, 5);
}
)cpp";

  EXPECT_TRUE(visitorCallbackLogEqual(
      RecordingVisitor(ShouldTraversePostOrder::No), Code,
      R"txt(
WalkUpFromStmt CompoundStmt
WalkUpFromStmt IntegerLiteral(1)
WalkUpFromStmt BinaryOperator(+)
WalkUpFromStmt IntegerLiteral(2)
WalkUpFromStmt IntegerLiteral(3)
TraverseCallExpr CallExpr(add)
  WalkUpFromStmt CallExpr(add)
  WalkUpFromStmt ImplicitCastExpr
  WalkUpFromStmt DeclRefExpr(add)
  WalkUpFromStmt IntegerLiteral(4)
  WalkUpFromStmt IntegerLiteral(5)
)txt"));

  EXPECT_TRUE(visitorCallbackLogEqual(
      RecordingVisitor(ShouldTraversePostOrder::Yes), Code,
      R"txt(
WalkUpFromStmt IntegerLiteral(1)
WalkUpFromStmt IntegerLiteral(2)
WalkUpFromStmt IntegerLiteral(3)
WalkUpFromStmt BinaryOperator(+)
TraverseCallExpr CallExpr(add)
  WalkUpFromStmt DeclRefExpr(add)
  WalkUpFromStmt ImplicitCastExpr
  WalkUpFromStmt IntegerLiteral(4)
  WalkUpFromStmt IntegerLiteral(5)
  WalkUpFromStmt CallExpr(add)
WalkUpFromStmt CompoundStmt
)txt"));
}

TEST(RecursiveASTVisitor, StmtCallbacks_TraverseCallExpr_WalkUpFromCallExpr) {
  class RecordingVisitor : public RecordingVisitorBase<RecordingVisitor> {
  public:
    RecordingVisitor(ShouldTraversePostOrder ShouldTraversePostOrderValue)
        : RecordingVisitorBase(ShouldTraversePostOrderValue) {}

    bool TraverseCallExpr(CallExpr *CE) {
      recordCallback(__func__, CE,
                     [&]() { RecordingVisitorBase::TraverseCallExpr(CE); });
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

    bool WalkUpFromCallExpr(CallExpr *CE) {
      recordCallback(__func__, CE,
                     [&]() { RecordingVisitorBase::WalkUpFromCallExpr(CE); });
      return true;
    }
  };

  StringRef Code = R"cpp(
void add(int, int);
void test() {
  1;
  2 + 3;
  add(4, 5);
}
)cpp";

  EXPECT_TRUE(visitorCallbackLogEqual(
      RecordingVisitor(ShouldTraversePostOrder::No), Code,
      R"txt(
WalkUpFromStmt CompoundStmt
WalkUpFromExpr IntegerLiteral(1)
  WalkUpFromStmt IntegerLiteral(1)
WalkUpFromExpr BinaryOperator(+)
  WalkUpFromStmt BinaryOperator(+)
WalkUpFromExpr IntegerLiteral(2)
  WalkUpFromStmt IntegerLiteral(2)
WalkUpFromExpr IntegerLiteral(3)
  WalkUpFromStmt IntegerLiteral(3)
TraverseCallExpr CallExpr(add)
  WalkUpFromCallExpr CallExpr(add)
    WalkUpFromExpr CallExpr(add)
      WalkUpFromStmt CallExpr(add)
  WalkUpFromExpr ImplicitCastExpr
    WalkUpFromStmt ImplicitCastExpr
  WalkUpFromExpr DeclRefExpr(add)
    WalkUpFromStmt DeclRefExpr(add)
  WalkUpFromExpr IntegerLiteral(4)
    WalkUpFromStmt IntegerLiteral(4)
  WalkUpFromExpr IntegerLiteral(5)
    WalkUpFromStmt IntegerLiteral(5)
)txt"));

  EXPECT_TRUE(visitorCallbackLogEqual(
      RecordingVisitor(ShouldTraversePostOrder::Yes), Code,
      R"txt(
WalkUpFromExpr IntegerLiteral(1)
  WalkUpFromStmt IntegerLiteral(1)
WalkUpFromExpr IntegerLiteral(2)
  WalkUpFromStmt IntegerLiteral(2)
WalkUpFromExpr IntegerLiteral(3)
  WalkUpFromStmt IntegerLiteral(3)
WalkUpFromExpr BinaryOperator(+)
  WalkUpFromStmt BinaryOperator(+)
TraverseCallExpr CallExpr(add)
  WalkUpFromExpr DeclRefExpr(add)
    WalkUpFromStmt DeclRefExpr(add)
  WalkUpFromExpr ImplicitCastExpr
    WalkUpFromStmt ImplicitCastExpr
  WalkUpFromExpr IntegerLiteral(4)
    WalkUpFromStmt IntegerLiteral(4)
  WalkUpFromExpr IntegerLiteral(5)
    WalkUpFromStmt IntegerLiteral(5)
  WalkUpFromCallExpr CallExpr(add)
    WalkUpFromExpr CallExpr(add)
      WalkUpFromStmt CallExpr(add)
WalkUpFromStmt CompoundStmt
)txt"));
}

TEST(RecursiveASTVisitor, StmtCallbacks_WalkUpFromCallExpr) {
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

    bool WalkUpFromCallExpr(CallExpr *CE) {
      recordCallback(__func__, CE,
                     [&]() { RecordingVisitorBase::WalkUpFromCallExpr(CE); });
      return true;
    }
  };

  StringRef Code = R"cpp(
void add(int, int);
void test() {
  1;
  2 + 3;
  add(4, 5);
}
)cpp";

  EXPECT_TRUE(visitorCallbackLogEqual(
      RecordingVisitor(ShouldTraversePostOrder::No), Code,
      R"txt(
WalkUpFromStmt CompoundStmt
WalkUpFromExpr IntegerLiteral(1)
  WalkUpFromStmt IntegerLiteral(1)
WalkUpFromExpr BinaryOperator(+)
  WalkUpFromStmt BinaryOperator(+)
WalkUpFromExpr IntegerLiteral(2)
  WalkUpFromStmt IntegerLiteral(2)
WalkUpFromExpr IntegerLiteral(3)
  WalkUpFromStmt IntegerLiteral(3)
WalkUpFromCallExpr CallExpr(add)
  WalkUpFromExpr CallExpr(add)
    WalkUpFromStmt CallExpr(add)
WalkUpFromExpr ImplicitCastExpr
  WalkUpFromStmt ImplicitCastExpr
WalkUpFromExpr DeclRefExpr(add)
  WalkUpFromStmt DeclRefExpr(add)
WalkUpFromExpr IntegerLiteral(4)
  WalkUpFromStmt IntegerLiteral(4)
WalkUpFromExpr IntegerLiteral(5)
  WalkUpFromStmt IntegerLiteral(5)
)txt"));

  EXPECT_TRUE(visitorCallbackLogEqual(
      RecordingVisitor(ShouldTraversePostOrder::Yes), Code,
      R"txt(
WalkUpFromExpr IntegerLiteral(1)
  WalkUpFromStmt IntegerLiteral(1)
WalkUpFromExpr IntegerLiteral(2)
  WalkUpFromStmt IntegerLiteral(2)
WalkUpFromExpr IntegerLiteral(3)
  WalkUpFromStmt IntegerLiteral(3)
WalkUpFromExpr BinaryOperator(+)
  WalkUpFromStmt BinaryOperator(+)
WalkUpFromExpr DeclRefExpr(add)
  WalkUpFromStmt DeclRefExpr(add)
WalkUpFromExpr ImplicitCastExpr
  WalkUpFromStmt ImplicitCastExpr
WalkUpFromExpr IntegerLiteral(4)
  WalkUpFromStmt IntegerLiteral(4)
WalkUpFromExpr IntegerLiteral(5)
  WalkUpFromStmt IntegerLiteral(5)
WalkUpFromCallExpr CallExpr(add)
  WalkUpFromExpr CallExpr(add)
    WalkUpFromStmt CallExpr(add)
WalkUpFromStmt CompoundStmt
)txt"));
}
