//===--- InefficientCopyAssignCheck.cpp - clang-tidy
//-------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InefficientCopyAssignCheck.h"

#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/Analyses/CFGReachabilityAnalysis.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/STLExtras.h"

using namespace clang::ast_matchers;

namespace clang::tidy::performance {

void InefficientCopyAssignCheck::registerMatchers(MatchFinder *Finder) {
  auto AssignOperatorExpr =
      cxxOperatorCallExpr(
          hasOperatorName("="),
          hasArgument(
              0, hasType(cxxRecordDecl(hasMethod(isMoveAssignmentOperator()))
                             .bind("assign-target-type"))),
          hasArgument(1, declRefExpr(to(varDecl().bind("assign-value-decl")))
                             .bind("assign-value")),
          hasAncestor(functionDecl().bind("within-func")))
          .bind("assign");
  Finder->addMatcher(AssignOperatorExpr, this);
}

CFG *InefficientCopyAssignCheck::getCFG(const FunctionDecl *FD,
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

void InefficientCopyAssignCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *AssignExpr = Result.Nodes.getNodeAs<Expr>("assign");
  const auto *AssignValue = Result.Nodes.getNodeAs<DeclRefExpr>("assign-value");
  const auto *AssignValueDecl =
      Result.Nodes.getNodeAs<VarDecl>("assign-value-decl");
  const auto *AssignTargetType =
      Result.Nodes.getNodeAs<CXXRecordDecl>("assign-target-type");
  const auto *WithinFunctionDecl =
      Result.Nodes.getNodeAs<FunctionDecl>("within-func");

  const QualType AssignValueQual = AssignValueDecl->getType();
  if (AssignValueQual->isReferenceType() ||
      AssignValueQual.isConstQualified() || AssignValueQual->isPointerType() ||
      AssignValueQual->isScalarType())
    return;

  if (AssignTargetType->hasTrivialMoveAssignment())
    return;

  if (CFG *TheCFG = getCFG(WithinFunctionDecl, Result.Context)) {
    // Walk the CFG bottom-up, starting with the exit node.
    // TODO: traverse the whole CFG instead of only considering terminator
    // nodes.

    CFGBlock &TheExit = TheCFG->getExit();
    for (auto &Pred : TheExit.preds()) {
      for (const CFGElement &Elt : llvm::reverse(*Pred)) {
        if (Elt.getKind() == CFGElement::Kind::Statement) {
          const Stmt *EltStmt = Elt.castAs<CFGStmt>().getStmt();
          if (EltStmt == AssignExpr) {
            diag(AssignValue->getBeginLoc(), "'%0' could be moved here")
                << AssignValue->getDecl()->getName();
            break;
          }
          // The reference is being referenced before the assignment, bail out.
          auto DeclRefMatcher =
              declRefExpr(hasDeclaration(equalsNode(AssignValue->getDecl())));
          if (!match(findAll(DeclRefMatcher), *EltStmt, *Result.Context)
                   .empty())
            break;
        }
      }
    }
  }
}

} // namespace clang::tidy::performance
