//===- unittests/Tooling/RecursiveASTVisitorTests/CallbacksUnaryOperator.cpp -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallbacksCommon.h"

TEST(RecursiveASTVisitor, StmtCallbacks_TraverseUnaryOperator) {
  class RecordingVisitor : public RecordingVisitorBase<RecordingVisitor> {
  public:
    RecordingVisitor(ShouldTraversePostOrder ShouldTraversePostOrderValue)
        : RecordingVisitorBase(ShouldTraversePostOrderValue) {}

    bool TraverseUnaryOperator(UnaryOperator *UO) {
      recordCallback(__func__, UO, [&]() {
        RecordingVisitorBase::TraverseUnaryOperator(UO);
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
void test() {
  1;
  -2;
  3;
}
)cpp";

  EXPECT_TRUE(visitorCallbackLogEqual(
      RecordingVisitor(ShouldTraversePostOrder::No), Code,
      R"txt(
WalkUpFromStmt CompoundStmt
WalkUpFromStmt IntegerLiteral(1)
TraverseUnaryOperator UnaryOperator(-)
  WalkUpFromStmt UnaryOperator(-)
  WalkUpFromStmt IntegerLiteral(2)
WalkUpFromStmt IntegerLiteral(3)
)txt"));

  EXPECT_TRUE(visitorCallbackLogEqual(
      RecordingVisitor(ShouldTraversePostOrder::Yes), Code,
      R"txt(
WalkUpFromStmt IntegerLiteral(1)
TraverseUnaryOperator UnaryOperator(-)
  WalkUpFromStmt IntegerLiteral(2)
  WalkUpFromStmt UnaryOperator(-)
WalkUpFromStmt IntegerLiteral(3)
WalkUpFromStmt CompoundStmt
)txt"));
}

TEST(RecursiveASTVisitor,
     StmtCallbacks_TraverseUnaryOperator_WalkUpFromUnaryOperator) {
  class RecordingVisitor : public RecordingVisitorBase<RecordingVisitor> {
  public:
    RecordingVisitor(ShouldTraversePostOrder ShouldTraversePostOrderValue)
        : RecordingVisitorBase(ShouldTraversePostOrderValue) {}

    bool TraverseUnaryOperator(UnaryOperator *UO) {
      recordCallback(__func__, UO, [&]() {
        RecordingVisitorBase::TraverseUnaryOperator(UO);
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

    bool WalkUpFromUnaryOperator(UnaryOperator *UO) {
      recordCallback(__func__, UO, [&]() {
        RecordingVisitorBase::WalkUpFromUnaryOperator(UO);
      });
      return true;
    }
  };

  StringRef Code = R"cpp(
void test() {
  1;
  -2;
  3;
}
)cpp";

  EXPECT_TRUE(visitorCallbackLogEqual(
      RecordingVisitor(ShouldTraversePostOrder::No), Code,
      R"txt(
WalkUpFromStmt CompoundStmt
WalkUpFromExpr IntegerLiteral(1)
  WalkUpFromStmt IntegerLiteral(1)
TraverseUnaryOperator UnaryOperator(-)
  WalkUpFromUnaryOperator UnaryOperator(-)
    WalkUpFromExpr UnaryOperator(-)
      WalkUpFromStmt UnaryOperator(-)
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
TraverseUnaryOperator UnaryOperator(-)
  WalkUpFromExpr IntegerLiteral(2)
    WalkUpFromStmt IntegerLiteral(2)
  WalkUpFromUnaryOperator UnaryOperator(-)
    WalkUpFromExpr UnaryOperator(-)
      WalkUpFromStmt UnaryOperator(-)
WalkUpFromExpr IntegerLiteral(3)
  WalkUpFromStmt IntegerLiteral(3)
WalkUpFromStmt CompoundStmt
)txt"));
}

TEST(RecursiveASTVisitor, StmtCallbacks_WalkUpFromUnaryOperator) {
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

    bool WalkUpFromUnaryOperator(UnaryOperator *UO) {
      recordCallback(__func__, UO, [&]() {
        RecordingVisitorBase::WalkUpFromUnaryOperator(UO);
      });
      return true;
    }
  };

  StringRef Code = R"cpp(
void test() {
  1;
  -2;
  3;
}
)cpp";

  EXPECT_TRUE(visitorCallbackLogEqual(
      RecordingVisitor(ShouldTraversePostOrder::No), Code,
      R"txt(
WalkUpFromStmt CompoundStmt
WalkUpFromExpr IntegerLiteral(1)
  WalkUpFromStmt IntegerLiteral(1)
WalkUpFromUnaryOperator UnaryOperator(-)
  WalkUpFromExpr UnaryOperator(-)
    WalkUpFromStmt UnaryOperator(-)
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
WalkUpFromExpr IntegerLiteral(2)
  WalkUpFromStmt IntegerLiteral(2)
WalkUpFromUnaryOperator UnaryOperator(-)
  WalkUpFromExpr UnaryOperator(-)
    WalkUpFromStmt UnaryOperator(-)
WalkUpFromExpr IntegerLiteral(3)
  WalkUpFromStmt IntegerLiteral(3)
WalkUpFromStmt CompoundStmt
)txt"));
}
