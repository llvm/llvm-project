//===--- UseStdMoveCheck.cpp - clang-tidy ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseStdMoveCheck.h"

#include "../utils/DeclRefExprUtils.h"

#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/Analyses/CFGReachabilityAnalysis.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/STLExtras.h"

using namespace clang::ast_matchers;

namespace clang::tidy::performance {

namespace {
AST_MATCHER(CXXRecordDecl, hasNonTrivialMoveAssignment) {
  return Node.hasNonTrivialMoveAssignment();
}

AST_MATCHER(QualType, isLValueReferenceType) {
  return Node->isLValueReferenceType();
}

AST_MATCHER(DeclRefExpr, refersToEnclosingVariableOrCapture) {
  return Node.refersToEnclosingVariableOrCapture();
}

AST_MATCHER(CXXOperatorCallExpr, isCopyAssignmentOperator) {
  if (const auto *MD = dyn_cast_or_null<CXXMethodDecl>(Node.getDirectCallee()))
    return MD->isCopyAssignmentOperator();
  return false;
}

// Ignore nodes inside macros.
AST_POLYMORPHIC_MATCHER(isInMacro,
                        AST_POLYMORPHIC_SUPPORTED_TYPES(Stmt, Decl)) {
  return Node.getBeginLoc().isMacroID() || Node.getEndLoc().isMacroID();
}
} // namespace

using utils::decl_ref_expr::allDeclRefExprs;

void UseStdMoveCheck::registerMatchers(MatchFinder *Finder) {
  auto AssignOperatorExpr =
      cxxOperatorCallExpr(
          isCopyAssignmentOperator(),
          hasArgument(0, hasType(cxxRecordDecl(hasNonTrivialMoveAssignment()))),
          hasArgument(
              1, declRefExpr(
                     to(varDecl(
                         hasLocalStorage(),
                         hasType(qualType(unless(anyOf(
                             isLValueReferenceType(),
                             isConstQualified() // Not valid to move const obj.
                             )))))),
                     unless(refersToEnclosingVariableOrCapture()))
                     .bind("assign-value")),
          forCallable(functionDecl().bind("within-func")), unless(isInMacro()))
          .bind("assign");
  Finder->addMatcher(AssignOperatorExpr, this);
}

const CFG *UseStdMoveCheck::getCFG(const FunctionDecl *FD,
                                   ASTContext *Context) {
  std::unique_ptr<CFG> &TheCFG = CFGCache[FD];
  if (!TheCFG) {
    const CFG::BuildOptions Options;
    std::unique_ptr<CFG> FCFG =
        CFG::buildCFG(nullptr, FD->getBody(), Context, Options);
    if (!FCFG)
      return nullptr;
    TheCFG.swap(FCFG);
  }
  return TheCFG.get();
}

void UseStdMoveCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *AssignExpr = Result.Nodes.getNodeAs<Expr>("assign");
  const auto *AssignValue = Result.Nodes.getNodeAs<DeclRefExpr>("assign-value");
  const auto *WithinFunctionDecl =
      Result.Nodes.getNodeAs<FunctionDecl>("within-func");

  const CFG *TheCFG = getCFG(WithinFunctionDecl, Result.Context);
  if (!TheCFG)
    return;

  // Walk the CFG bottom-up, starting with the exit node.
  // TODO: traverse the whole CFG instead of only considering terminator
  // nodes.
  const CFGBlock &TheExit = TheCFG->getExit();
  for (auto &Pred : TheExit.preds()) {
    if (!Pred.isReachable())
      continue;
    for (const CFGElement &Elt : llvm::reverse(*Pred)) {
      if (Elt.getKind() != CFGElement::Kind::Statement)
        continue;

      const Stmt *EltStmt = Elt.castAs<CFGStmt>().getStmt();
      if (EltStmt == AssignExpr) {
        diag(AssignValue->getBeginLoc(), "'%0' could be moved here")
            << AssignValue->getDecl()->getName();
        break;
      }
      // The reference is being referenced after the assignment, bail out.
      if (!allDeclRefExprs(*cast<VarDecl>(AssignValue->getDecl()), *EltStmt,
                           *Result.Context)
               .empty())
        break;
    }
  }
}

} // namespace clang::tidy::performance
