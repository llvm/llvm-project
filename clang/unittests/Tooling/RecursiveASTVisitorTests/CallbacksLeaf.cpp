//===--- unittests/Tooling/RecursiveASTVisitorTests/CallbacksLeaf.cpp -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallbacksCommon.h"

TEST(RecursiveASTVisitor, StmtCallbacks_TraverseLeaf) {
  class RecordingVisitor : public RecordingVisitorBase<RecordingVisitor> {
  public:
    RecordingVisitor(ShouldTraversePostOrder ShouldTraversePostOrderValue)
        : RecordingVisitorBase(ShouldTraversePostOrderValue) {}

    bool TraverseIntegerLiteral(IntegerLiteral *IL) {
      recordCallback(__func__, IL, [&]() {
        RecordingVisitorBase::TraverseIntegerLiteral(IL);
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
TraverseIntegerLiteral IntegerLiteral(1)
  WalkUpFromStmt IntegerLiteral(1)
WalkUpFromStmt BinaryOperator(+)
TraverseIntegerLiteral IntegerLiteral(2)
  WalkUpFromStmt IntegerLiteral(2)
TraverseIntegerLiteral IntegerLiteral(3)
  WalkUpFromStmt IntegerLiteral(3)
WalkUpFromStmt CallExpr(add)
WalkUpFromStmt ImplicitCastExpr
WalkUpFromStmt DeclRefExpr(add)
TraverseIntegerLiteral IntegerLiteral(4)
  WalkUpFromStmt IntegerLiteral(4)
TraverseIntegerLiteral IntegerLiteral(5)
  WalkUpFromStmt IntegerLiteral(5)
)txt"));

  EXPECT_TRUE(visitorCallbackLogEqual(
      RecordingVisitor(ShouldTraversePostOrder::Yes), Code,
      R"txt(
TraverseIntegerLiteral IntegerLiteral(1)
  WalkUpFromStmt IntegerLiteral(1)
TraverseIntegerLiteral IntegerLiteral(2)
  WalkUpFromStmt IntegerLiteral(2)
TraverseIntegerLiteral IntegerLiteral(3)
  WalkUpFromStmt IntegerLiteral(3)
WalkUpFromStmt BinaryOperator(+)
WalkUpFromStmt DeclRefExpr(add)
WalkUpFromStmt ImplicitCastExpr
TraverseIntegerLiteral IntegerLiteral(4)
  WalkUpFromStmt IntegerLiteral(4)
TraverseIntegerLiteral IntegerLiteral(5)
  WalkUpFromStmt IntegerLiteral(5)
WalkUpFromStmt CallExpr(add)
WalkUpFromStmt CompoundStmt
)txt"));
}

TEST(RecursiveASTVisitor, StmtCallbacks_TraverseLeaf_WalkUpFromLeaf) {
  class RecordingVisitor : public RecordingVisitorBase<RecordingVisitor> {
  public:
    RecordingVisitor(ShouldTraversePostOrder ShouldTraversePostOrderValue)
        : RecordingVisitorBase(ShouldTraversePostOrderValue) {}

    bool TraverseIntegerLiteral(IntegerLiteral *IL) {
      recordCallback(__func__, IL, [&]() {
        RecordingVisitorBase::TraverseIntegerLiteral(IL);
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

    bool WalkUpFromIntegerLiteral(IntegerLiteral *IL) {
      recordCallback(__func__, IL, [&]() {
        RecordingVisitorBase::WalkUpFromIntegerLiteral(IL);
      });
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
TraverseIntegerLiteral IntegerLiteral(1)
  WalkUpFromIntegerLiteral IntegerLiteral(1)
    WalkUpFromExpr IntegerLiteral(1)
      WalkUpFromStmt IntegerLiteral(1)
WalkUpFromExpr BinaryOperator(+)
  WalkUpFromStmt BinaryOperator(+)
TraverseIntegerLiteral IntegerLiteral(2)
  WalkUpFromIntegerLiteral IntegerLiteral(2)
    WalkUpFromExpr IntegerLiteral(2)
      WalkUpFromStmt IntegerLiteral(2)
TraverseIntegerLiteral IntegerLiteral(3)
  WalkUpFromIntegerLiteral IntegerLiteral(3)
    WalkUpFromExpr IntegerLiteral(3)
      WalkUpFromStmt IntegerLiteral(3)
WalkUpFromExpr CallExpr(add)
  WalkUpFromStmt CallExpr(add)
WalkUpFromExpr ImplicitCastExpr
  WalkUpFromStmt ImplicitCastExpr
WalkUpFromExpr DeclRefExpr(add)
  WalkUpFromStmt DeclRefExpr(add)
TraverseIntegerLiteral IntegerLiteral(4)
  WalkUpFromIntegerLiteral IntegerLiteral(4)
    WalkUpFromExpr IntegerLiteral(4)
      WalkUpFromStmt IntegerLiteral(4)
TraverseIntegerLiteral IntegerLiteral(5)
  WalkUpFromIntegerLiteral IntegerLiteral(5)
    WalkUpFromExpr IntegerLiteral(5)
      WalkUpFromStmt IntegerLiteral(5)
)txt"));

  EXPECT_TRUE(visitorCallbackLogEqual(
      RecordingVisitor(ShouldTraversePostOrder::Yes), Code,
      R"txt(
TraverseIntegerLiteral IntegerLiteral(1)
  WalkUpFromIntegerLiteral IntegerLiteral(1)
    WalkUpFromExpr IntegerLiteral(1)
      WalkUpFromStmt IntegerLiteral(1)
TraverseIntegerLiteral IntegerLiteral(2)
  WalkUpFromIntegerLiteral IntegerLiteral(2)
    WalkUpFromExpr IntegerLiteral(2)
      WalkUpFromStmt IntegerLiteral(2)
TraverseIntegerLiteral IntegerLiteral(3)
  WalkUpFromIntegerLiteral IntegerLiteral(3)
    WalkUpFromExpr IntegerLiteral(3)
      WalkUpFromStmt IntegerLiteral(3)
WalkUpFromExpr BinaryOperator(+)
  WalkUpFromStmt BinaryOperator(+)
WalkUpFromExpr DeclRefExpr(add)
  WalkUpFromStmt DeclRefExpr(add)
WalkUpFromExpr ImplicitCastExpr
  WalkUpFromStmt ImplicitCastExpr
TraverseIntegerLiteral IntegerLiteral(4)
  WalkUpFromIntegerLiteral IntegerLiteral(4)
    WalkUpFromExpr IntegerLiteral(4)
      WalkUpFromStmt IntegerLiteral(4)
TraverseIntegerLiteral IntegerLiteral(5)
  WalkUpFromIntegerLiteral IntegerLiteral(5)
    WalkUpFromExpr IntegerLiteral(5)
      WalkUpFromStmt IntegerLiteral(5)
WalkUpFromExpr CallExpr(add)
  WalkUpFromStmt CallExpr(add)
WalkUpFromStmt CompoundStmt
)txt"));
}

TEST(RecursiveASTVisitor, StmtCallbacks_WalkUpFromLeaf) {
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

    bool WalkUpFromIntegerLiteral(IntegerLiteral *IL) {
      recordCallback(__func__, IL, [&]() {
        RecordingVisitorBase::WalkUpFromIntegerLiteral(IL);
      });
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
WalkUpFromIntegerLiteral IntegerLiteral(1)
  WalkUpFromExpr IntegerLiteral(1)
    WalkUpFromStmt IntegerLiteral(1)
WalkUpFromExpr BinaryOperator(+)
  WalkUpFromStmt BinaryOperator(+)
WalkUpFromIntegerLiteral IntegerLiteral(2)
  WalkUpFromExpr IntegerLiteral(2)
    WalkUpFromStmt IntegerLiteral(2)
WalkUpFromIntegerLiteral IntegerLiteral(3)
  WalkUpFromExpr IntegerLiteral(3)
    WalkUpFromStmt IntegerLiteral(3)
WalkUpFromExpr CallExpr(add)
  WalkUpFromStmt CallExpr(add)
WalkUpFromExpr ImplicitCastExpr
  WalkUpFromStmt ImplicitCastExpr
WalkUpFromExpr DeclRefExpr(add)
  WalkUpFromStmt DeclRefExpr(add)
WalkUpFromIntegerLiteral IntegerLiteral(4)
  WalkUpFromExpr IntegerLiteral(4)
    WalkUpFromStmt IntegerLiteral(4)
WalkUpFromIntegerLiteral IntegerLiteral(5)
  WalkUpFromExpr IntegerLiteral(5)
    WalkUpFromStmt IntegerLiteral(5)
)txt"));

  EXPECT_TRUE(visitorCallbackLogEqual(
      RecordingVisitor(ShouldTraversePostOrder::Yes), Code,
      R"txt(
WalkUpFromIntegerLiteral IntegerLiteral(1)
  WalkUpFromExpr IntegerLiteral(1)
    WalkUpFromStmt IntegerLiteral(1)
WalkUpFromIntegerLiteral IntegerLiteral(2)
  WalkUpFromExpr IntegerLiteral(2)
    WalkUpFromStmt IntegerLiteral(2)
WalkUpFromIntegerLiteral IntegerLiteral(3)
  WalkUpFromExpr IntegerLiteral(3)
    WalkUpFromStmt IntegerLiteral(3)
WalkUpFromExpr BinaryOperator(+)
  WalkUpFromStmt BinaryOperator(+)
WalkUpFromExpr DeclRefExpr(add)
  WalkUpFromStmt DeclRefExpr(add)
WalkUpFromExpr ImplicitCastExpr
  WalkUpFromStmt ImplicitCastExpr
WalkUpFromIntegerLiteral IntegerLiteral(4)
  WalkUpFromExpr IntegerLiteral(4)
    WalkUpFromStmt IntegerLiteral(4)
WalkUpFromIntegerLiteral IntegerLiteral(5)
  WalkUpFromExpr IntegerLiteral(5)
    WalkUpFromStmt IntegerLiteral(5)
WalkUpFromExpr CallExpr(add)
  WalkUpFromStmt CallExpr(add)
WalkUpFromStmt CompoundStmt
)txt"));
}
