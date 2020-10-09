//===--- unittests/Tooling/RecursiveASTVisitorTests/CallbacksCommon.h -----===//
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
    if (UnaryOperator *UO = dyn_cast<UnaryOperator>(S)) {
      return (ClassName + "(" + UnaryOperator::getOpcodeStr(UO->getOpcode()) +
              ")")
          .str();
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
