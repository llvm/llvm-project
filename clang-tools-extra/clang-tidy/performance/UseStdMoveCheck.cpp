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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

using namespace clang::ast_matchers;

namespace clang::tidy::performance {

namespace {
AST_MATCHER(CXXRecordDecl, hasAccessibleNonTrivialMoveAssignment) {
  if (!Node.hasNonTrivialMoveAssignment())
    return false;
  for (const auto *CM : Node.methods())
    if (CM->isMoveAssignmentOperator())
      return !CM->isDeleted() && CM->getAccess() == AS_public;
  llvm_unreachable("Move Assignment Operator Not Found");
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
          hasArgument(0, hasType(cxxRecordDecl(
                             hasAccessibleNonTrivialMoveAssignment()))),
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

  // The algorithm to look for a convertible move-assign operator is the
  // following: each node starts in the `Ready` state, with a number of
  // `RemainingSuccessors` equal to its number of successors.
  //
  // Starting from the exit node, we walk the CFG backward. Whenever
  // we meet a new block, we check if it either:
  // 1. touches the `AssignValue`, in which case we stop the search, and mark
  // each
  //    predecessor as not `Ready`. No predecessor walk.
  // 2. contains a convertible copy-assign operator, in which case we generate a
  //    fix, and mark each predecessor as not Ready. No predecessor walk.
  // 3. does not interact with `AssignValue`, in which case we decrement the
  //    `RemainingSuccessors` of each predecessor. And if it happens to turn to
  //    0 while still being `Ready`, we add it to the `WorkList`.

  struct BlockState {
    bool Ready;
    unsigned RemainingSuccessors;
  };
  llvm::DenseMap<const CFGBlock *, BlockState> CFGState;
  for (const auto *B : *TheCFG)
    CFGState.try_emplace(B, BlockState{true, B->succ_size()});

  const CFGBlock &TheExit = TheCFG->getExit();
  std::vector<const CFGBlock *> WorkList = {&TheExit};

  while (!WorkList.empty()) {
    const CFGBlock *B = WorkList.back();
    WorkList.pop_back();
    const BlockState &BS = CFGState.find(B)->second;
    if (!BS.Ready)
      continue;

    assert(BS.RemainingSuccessors == 0 &&
           "All successors have been processed.");
    bool ReferencesAssignedValue = false;
    for (const CFGElement &Elt : llvm::reverse(*B)) {
      if (Elt.getKind() != CFGElement::Kind::Statement)
        continue;

      const Stmt *EltStmt = Elt.castAs<CFGStmt>().getStmt();
      if (EltStmt == AssignExpr) {
        const StringRef AssignValueName = AssignValue->getDecl()->getName();
        diag(AssignValue->getBeginLoc(), "'%0' could be moved here")
            << AssignValueName
            << FixItHint::CreateReplacement(
                   AssignValue->getLocation(),
                   ("std::move(" + AssignValueName + ")").str());
        ReferencesAssignedValue = true;
        break;
      }

      // The reference is being referenced after the assignment.
      if (!allDeclRefExprs(*cast<VarDecl>(AssignValue->getDecl()), *EltStmt,
                           *Result.Context)
               .empty()) {
        ReferencesAssignedValue = true;
        break;
      }
    }
    if (ReferencesAssignedValue) {
      // Cancel all predecessors.
      for (const auto &S : B->preds()) {
        if (!S.isReachable())
          continue;
        CFGState.find(&*S)->second.Ready = false;
      }
    } else {
      // Or process the ready ones.
      for (const auto &S : B->preds()) {
        if (!S.isReachable())
          continue;
        auto &W = CFGState.find(&*S)->second;
        if (W.Ready) {
          if (--W.RemainingSuccessors == 0 && S.isReachable())
            WorkList.push_back(&*S);
        }
      }
    }
  }
}

} // namespace clang::tidy::performance
