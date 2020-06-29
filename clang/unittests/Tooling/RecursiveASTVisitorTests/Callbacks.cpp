//===--- clang/unittests/Tooling/RecursiveASTVisitorTests/Callbacks.cpp ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestVisitor.h"

using namespace clang;

namespace {

enum class ShouldTraversePostOrder : bool {
  No = false,
  Yes = true,
};

/// Base class for tests for RecursiveASTVisitor tests that validate the
/// sequence of calls to user-defined callbacks like Traverse*(), WalkUp*(),
/// Visit*().
template <typename Derived>
class RecordingVisitorBase : public TestVisitor<Derived> {
  ShouldTraversePostOrder ShouldTraversePostOrderValue;

public:
  RecordingVisitorBase(ShouldTraversePostOrder ShouldTraversePostOrderValue)
      : ShouldTraversePostOrderValue(ShouldTraversePostOrderValue) {}

  bool shouldTraversePostOrder() const {
    return static_cast<bool>(ShouldTraversePostOrderValue);
  }

  // Callbacks received during traversal.
  std::string CallbackLog;
  unsigned CallbackLogIndent = 0;

  std::string stmtToString(Stmt *S) {
    StringRef ClassName = S->getStmtClassName();
    if (IntegerLiteral *IL = dyn_cast<IntegerLiteral>(S)) {
      return (ClassName + "(" + IL->getValue().toString(10, false) + ")").str();
    }
    if (BinaryOperator *BO = dyn_cast<BinaryOperator>(S)) {
      return (ClassName + "(" + BinaryOperator::getOpcodeStr(BO->getOpcode()) +
              ")")
          .str();
    }
    if (CallExpr *CE = dyn_cast<CallExpr>(S)) {
      if (FunctionDecl *Callee = CE->getDirectCallee()) {
        if (Callee->getIdentifier()) {
          return (ClassName + "(" + Callee->getName() + ")").str();
        }
      }
    }
    if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(S)) {
      if (NamedDecl *ND = DRE->getFoundDecl()) {
        if (ND->getIdentifier()) {
          return (ClassName + "(" + ND->getName() + ")").str();
        }
      }
    }
    return ClassName.str();
  }

  /// Record the fact that the user-defined callback member function
  /// \p CallbackName was called with the argument \p S. Then, record the
  /// effects of calling the default implementation \p CallDefaultFn.
  template <typename CallDefault>
  void recordCallback(StringRef CallbackName, Stmt *S,
                      CallDefault CallDefaultFn) {
    for (unsigned i = 0; i != CallbackLogIndent; ++i) {
      CallbackLog += "  ";
    }
    CallbackLog += (CallbackName + " " + stmtToString(S) + "\n").str();
    ++CallbackLogIndent;
    CallDefaultFn();
    --CallbackLogIndent;
  }
};

template <typename VisitorTy>
::testing::AssertionResult visitorCallbackLogEqual(VisitorTy Visitor,
                                                   StringRef Code,
                                                   StringRef ExpectedLog) {
  Visitor.runOver(Code);
  // EXPECT_EQ shows the diff between the two strings if they are different.
  EXPECT_EQ(ExpectedLog.trim().str(),
            StringRef(Visitor.CallbackLog).trim().str());
  if (ExpectedLog.trim() != StringRef(Visitor.CallbackLog).trim()) {
    return ::testing::AssertionFailure();
  }
  return ::testing::AssertionSuccess();
}

} // namespace

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
WalkUpFromStmt IntegerLiteral(1)
TraverseIntegerLiteral IntegerLiteral(2)
  WalkUpFromStmt IntegerLiteral(2)
WalkUpFromStmt IntegerLiteral(2)
TraverseIntegerLiteral IntegerLiteral(3)
  WalkUpFromStmt IntegerLiteral(3)
WalkUpFromStmt IntegerLiteral(3)
WalkUpFromStmt BinaryOperator(+)
WalkUpFromStmt DeclRefExpr(add)
WalkUpFromStmt ImplicitCastExpr
TraverseIntegerLiteral IntegerLiteral(4)
  WalkUpFromStmt IntegerLiteral(4)
WalkUpFromStmt IntegerLiteral(4)
TraverseIntegerLiteral IntegerLiteral(5)
  WalkUpFromStmt IntegerLiteral(5)
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
WalkUpFromIntegerLiteral IntegerLiteral(1)
  WalkUpFromExpr IntegerLiteral(1)
    WalkUpFromStmt IntegerLiteral(1)
TraverseIntegerLiteral IntegerLiteral(2)
  WalkUpFromIntegerLiteral IntegerLiteral(2)
    WalkUpFromExpr IntegerLiteral(2)
      WalkUpFromStmt IntegerLiteral(2)
WalkUpFromIntegerLiteral IntegerLiteral(2)
  WalkUpFromExpr IntegerLiteral(2)
    WalkUpFromStmt IntegerLiteral(2)
TraverseIntegerLiteral IntegerLiteral(3)
  WalkUpFromIntegerLiteral IntegerLiteral(3)
    WalkUpFromExpr IntegerLiteral(3)
      WalkUpFromStmt IntegerLiteral(3)
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
WalkUpFromIntegerLiteral IntegerLiteral(4)
  WalkUpFromExpr IntegerLiteral(4)
    WalkUpFromStmt IntegerLiteral(4)
TraverseIntegerLiteral IntegerLiteral(5)
  WalkUpFromIntegerLiteral IntegerLiteral(5)
    WalkUpFromExpr IntegerLiteral(5)
      WalkUpFromStmt IntegerLiteral(5)
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
WalkUpFromStmt DeclRefExpr(add)
WalkUpFromStmt ImplicitCastExpr
WalkUpFromStmt IntegerLiteral(4)
WalkUpFromStmt IntegerLiteral(5)
WalkUpFromStmt CallExpr(add)
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
void add(int, int);
void test() {
  1;
  2 + 3;
  add(4, 5);
}
)cpp";

  // FIXME: It is arguably a bug in RecursiveASTVisitor that
  // WalkUpFromBinaryOperator is called, but TraverseBinaryOperator is not
  // called.
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
WalkUpFromBinaryOperator BinaryOperator(+)
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
WalkUpFromExpr CallExpr(add)
  WalkUpFromStmt CallExpr(add)
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
WalkUpFromBinaryOperator BinaryOperator(+)
  WalkUpFromExpr BinaryOperator(+)
    WalkUpFromStmt BinaryOperator(+)
WalkUpFromExpr IntegerLiteral(2)
  WalkUpFromStmt IntegerLiteral(2)
WalkUpFromExpr IntegerLiteral(3)
  WalkUpFromStmt IntegerLiteral(3)
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
WalkUpFromBinaryOperator BinaryOperator(+)
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
WalkUpFromExpr CallExpr(add)
  WalkUpFromStmt CallExpr(add)
WalkUpFromStmt CompoundStmt
)txt"));
}

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
