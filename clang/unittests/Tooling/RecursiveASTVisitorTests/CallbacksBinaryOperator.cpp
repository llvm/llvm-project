//===- unittests/Tooling/RecursiveASTVisitorTests/CallbacksBinaryOperator.cpp -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallbacksCommon.h"

TEST(RecursiveASTVisitor, StmtCallbacks_TraverseBinaryOperator) {
  class RecordingVisitor : public RecordingVisitorBase<RecordingVisitor> {
  public:
    RecordingVisitor(ShouldTraversePostOrder ShouldTraversePostOrderValue)
        : RecordingVisitorBase(ShouldTraversePostOrderValue) {}

    bool TraverseBinaryOperator(BinaryOperator *BO) {
      recordCallback(__func__, BO, [&]() {
        RecordingVisitorBase::TraverseBinaryOperator(BO);
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
  2 + 3;
  4;
}
)cpp";

  EXPECT_TRUE(visitorCallbackLogEqual(
      RecordingVisitor(ShouldTraversePostOrder::No), Code,
      R"txt(
WalkUpFromStmt CompoundStmt
WalkUpFromStmt IntegerLiteral(1)
TraverseBinaryOperator BinaryOperator(+)
  WalkUpFromStmt BinaryOperator(+)
  WalkUpFromStmt IntegerLiteral(2)
  WalkUpFromStmt IntegerLiteral(3)
WalkUpFromStmt IntegerLiteral(4)
)txt"));

  EXPECT_TRUE(visitorCallbackLogEqual(
      RecordingVisitor(ShouldTraversePostOrder::Yes), Code,
      R"txt(
WalkUpFromStmt IntegerLiteral(1)
TraverseBinaryOperator BinaryOperator(+)
  WalkUpFromStmt IntegerLiteral(2)
  WalkUpFromStmt IntegerLiteral(3)
  WalkUpFromStmt BinaryOperator(+)
WalkUpFromStmt IntegerLiteral(4)
WalkUpFromStmt CompoundStmt
)txt"));
}

TEST(RecursiveASTVisitor,
     StmtCallbacks_TraverseBinaryOperator_WalkUpFromBinaryOperator) {
  class RecordingVisitor : public RecordingVisitorBase<RecordingVisitor> {
  public:
    RecordingVisitor(ShouldTraversePostOrder ShouldTraversePostOrderValue)
        : RecordingVisitorBase(ShouldTraversePostOrderValue) {}

    bool TraverseBinaryOperator(BinaryOperator *BO) {
      recordCallback(__func__, BO, [&]() {
        RecordingVisitorBase::TraverseBinaryOperator(BO);
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

    bool WalkUpFromBinaryOperator(BinaryOperator *BO) {
      recordCallback(__func__, BO, [&]() {
        RecordingVisitorBase::WalkUpFromBinaryOperator(BO);
      });
      return true;
    }
  };

  StringRef Code = R"cpp(
void test() {
  1;
  2 + 3;
  4;
}
)cpp";

  EXPECT_TRUE(visitorCallbackLogEqual(
      RecordingVisitor(ShouldTraversePostOrder::No), Code,
      R"txt(
WalkUpFromStmt CompoundStmt
WalkUpFromExpr IntegerLiteral(1)
  WalkUpFromStmt IntegerLiteral(1)
TraverseBinaryOperator BinaryOperator(+)
  WalkUpFromBinaryOperator BinaryOperator(+)
    WalkUpFromExpr BinaryOperator(+)
      WalkUpFromStmt BinaryOperator(+)
  WalkUpFromExpr IntegerLiteral(2)
    WalkUpFromStmt IntegerLiteral(2)
  WalkUpFromExpr IntegerLiteral(3)
    WalkUpFromStmt IntegerLiteral(3)
WalkUpFromExpr IntegerLiteral(4)
  WalkUpFromStmt IntegerLiteral(4)
)txt"));

  EXPECT_TRUE(visitorCallbackLogEqual(
      RecordingVisitor(ShouldTraversePostOrder::Yes), Code,
      R"txt(
WalkUpFromExpr IntegerLiteral(1)
  WalkUpFromStmt IntegerLiteral(1)
TraverseBinaryOperator BinaryOperator(+)
  WalkUpFromExpr IntegerLiteral(2)
    WalkUpFromStmt IntegerLiteral(2)
  WalkUpFromExpr IntegerLiteral(3)
    WalkUpFromStmt IntegerLiteral(3)
  WalkUpFromBinaryOperator BinaryOperator(+)
    WalkUpFromExpr BinaryOperator(+)
      WalkUpFromStmt BinaryOperator(+)
WalkUpFromExpr IntegerLiteral(4)
  WalkUpFromStmt IntegerLiteral(4)
WalkUpFromStmt CompoundStmt
)txt"));
}

TEST(RecursiveASTVisitor, StmtCallbacks_WalkUpFromBinaryOperator) {
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

    bool WalkUpFromBinaryOperator(BinaryOperator *BO) {
      recordCallback(__func__, BO, [&]() {
        RecordingVisitorBase::WalkUpFromBinaryOperator(BO);
      });
      return true;
    }
  };

  StringRef Code = R"cpp(
void test() {
  1;
  2 + 3;
  4;
}
)cpp";

  EXPECT_TRUE(visitorCallbackLogEqual(
      RecordingVisitor(ShouldTraversePostOrder::No), Code,
      R"txt(
WalkUpFromStmt CompoundStmt
WalkUpFromExpr IntegerLiteral(1)
  WalkUpFromStmt IntegerLiteral(1)
WalkUpFromBinaryOperator BinaryOperator(+)
  WalkUpFromExpr BinaryOperator(+)
    WalkUpFromStmt BinaryOperator(+)
WalkUpFromExpr IntegerLiteral(2)
  WalkUpFromStmt IntegerLiteral(2)
WalkUpFromExpr IntegerLiteral(3)
  WalkUpFromStmt IntegerLiteral(3)
WalkUpFromExpr IntegerLiteral(4)
  WalkUpFromStmt IntegerLiteral(4)
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
WalkUpFromBinaryOperator BinaryOperator(+)
  WalkUpFromExpr BinaryOperator(+)
    WalkUpFromStmt BinaryOperator(+)
WalkUpFromExpr IntegerLiteral(4)
  WalkUpFromStmt IntegerLiteral(4)
WalkUpFromStmt CompoundStmt
)txt"));
}
