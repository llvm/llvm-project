//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SuspiciousReallocUsageCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;
using namespace clang;

namespace {
/// Check if two different expression nodes denote the same
/// "pointer expression". The "pointer expression" can consist of member
/// expressions and declaration references only (like \c a->b->c), otherwise the
/// check is always false.
class IsSamePtrExpr : public StmtVisitor<IsSamePtrExpr, bool> {
  /// The other expression to compare against.
  /// This variable is used to pass the data from a \c check function to any of
  /// the visit functions. Every visit function starts by converting \c OtherE
  /// to the current type and store it locally, and do not use \c OtherE later.
  const Expr *OtherE = nullptr;

public:
  bool VisitDeclRefExpr(const DeclRefExpr *E1) {
    const auto *E2 = dyn_cast<DeclRefExpr>(OtherE);
    if (!E2)
      return false;
    const Decl *D1 = E1->getDecl()->getCanonicalDecl();
    return isa<VarDecl, FieldDecl>(D1) &&
           D1 == E2->getDecl()->getCanonicalDecl();
  }

  bool VisitMemberExpr(const MemberExpr *E1) {
    const auto *E2 = dyn_cast<MemberExpr>(OtherE);
    if (!E2)
      return false;
    if (!check(E1->getBase(), E2->getBase()))
      return false;
    DeclAccessPair FD = E1->getFoundDecl();
    return isa<FieldDecl>(FD.getDecl()) && FD == E2->getFoundDecl();
  }

  bool check(const Expr *E1, const Expr *E2) {
    E1 = E1->IgnoreParenCasts();
    E2 = E2->IgnoreParenCasts();
    OtherE = E2;
    return Visit(const_cast<Expr *>(E1));
  }
};

/// Check if there is an assignment or initialization that references a variable
/// \c Var (at right-hand side) and is before \c VarRef in the source code.
/// Only simple assignments like \code a = b \endcode are found.
class FindAssignToVarBefore
    : public ConstStmtVisitor<FindAssignToVarBefore, bool> {
  const VarDecl *Var;
  const DeclRefExpr *VarRef;
  SourceManager &SM;

  bool isAccessForVar(const Expr *E) const {
    if (const auto *DeclRef = dyn_cast<DeclRefExpr>(E->IgnoreParenCasts()))
      return DeclRef->getDecl() &&
             DeclRef->getDecl()->getCanonicalDecl() == Var &&
             SM.isBeforeInTranslationUnit(E->getBeginLoc(),
                                          VarRef->getBeginLoc());
    return false;
  }

public:
  FindAssignToVarBefore(const VarDecl *Var, const DeclRefExpr *VarRef,
                        SourceManager &SM)
      : Var(Var->getCanonicalDecl()), VarRef(VarRef), SM(SM) {}

  bool VisitDeclStmt(const DeclStmt *S) {
    for (const Decl *D : S->getDeclGroup())
      if (const auto *LeftVar = dyn_cast<VarDecl>(D))
        if (LeftVar->hasInit())
          return isAccessForVar(LeftVar->getInit());
    return false;
  }
  bool VisitBinaryOperator(const BinaryOperator *S) {
    if (S->getOpcode() == BO_Assign)
      return isAccessForVar(S->getRHS());
    return false;
  }
  bool VisitStmt(const Stmt *S) {
    for (const Stmt *Child : S->children())
      if (Child && Visit(Child))
        return true;
    return false;
  }
};

} // namespace

namespace clang::tidy::bugprone {

void SuspiciousReallocUsageCheck::registerMatchers(MatchFinder *Finder) {
  // void *realloc(void *ptr, size_t size);
  auto ReallocDecl =
      functionDecl(hasName("::realloc"), parameterCountIs(2),
                   hasParameter(0, hasType(pointerType(pointee(voidType())))),
                   hasParameter(1, hasType(isInteger())))
          .bind("realloc");

  auto ReallocCall =
      callExpr(callee(ReallocDecl), hasArgument(0, expr().bind("ptr_input")),
               hasAncestor(functionDecl().bind("parent_function")))
          .bind("call");
  Finder->addMatcher(binaryOperator(hasOperatorName("="),
                                    hasLHS(expr().bind("ptr_result")),
                                    hasRHS(ignoringParenCasts(ReallocCall))),
                     this);
}

void SuspiciousReallocUsageCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Call = Result.Nodes.getNodeAs<CallExpr>("call");
  if (!Call)
    return;
  const auto *PtrInputExpr = Result.Nodes.getNodeAs<Expr>("ptr_input");
  const auto *PtrResultExpr = Result.Nodes.getNodeAs<Expr>("ptr_result");
  if (!PtrInputExpr || !PtrResultExpr)
    return;
  const auto *ReallocD = Result.Nodes.getNodeAs<Decl>("realloc");
  assert(ReallocD && "Value for 'realloc' should exist if 'call' was found.");
  SourceManager &SM = ReallocD->getASTContext().getSourceManager();

  if (!IsSamePtrExpr{}.check(PtrInputExpr, PtrResultExpr))
    return;

  if (const auto *DeclRef =
          dyn_cast<DeclRefExpr>(PtrInputExpr->IgnoreParenImpCasts()))
    if (const auto *Var = dyn_cast<VarDecl>(DeclRef->getDecl()))
      if (const auto *Func =
              Result.Nodes.getNodeAs<FunctionDecl>("parent_function"))
        if (FindAssignToVarBefore{Var, DeclRef, SM}.Visit(Func->getBody()))
          return;

  StringRef CodeOfAssignedExpr = Lexer::getSourceText(
      CharSourceRange::getTokenRange(PtrResultExpr->getSourceRange()), SM,
      getLangOpts());
  diag(Call->getBeginLoc(), "'%0' may be set to null if 'realloc' fails, which "
                            "may result in a leak of the original buffer")
      << CodeOfAssignedExpr << PtrInputExpr->getSourceRange()
      << PtrResultExpr->getSourceRange();
}

} // namespace clang::tidy::bugprone
