//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RethrowCaughtExceptionCheck.h"
#include "../utils/ASTUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/IgnoreExpr.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

// True when the throw operand spells an explicit construction, e.g.
// `throw Error(e);` or `throw Error{e};`. Only implicit wrappers are
// stripped, so anything else tests negative.
static bool isExplicitConstruction(const Expr *E) {
  E = IgnoreExprNodes(E, IgnoreParensSingleStep, IgnoreImplicitSingleStep,
                      IgnoreElidableImplicitConstructorSingleStep);
  return isa<CXXFunctionalCastExpr, CXXTemporaryObjectExpr>(E);
}

// Nearest CXXCatchStmt, LambdaExpr, or BlockExpr up the parent chain;
// nullptr when the chain ends, branches, or first reaches a function.
static const Stmt *findNearestScope(const CXXThrowExpr *Throw,
                                    ASTContext *Context) {
  DynTypedNode Node = DynTypedNode::create(*Throw);
  while (true) {
    const DynTypedNodeList Parents = Context->getParents(Node);
    // Ambiguous parents: prefer a false negative over a false positive.
    if (Parents.size() != 1)
      return nullptr;
    const DynTypedNode Parent = Parents[0];
    if (const auto *Scope = Parent.get<Stmt>()) {
      if (isa<CXXCatchStmt, LambdaExpr, BlockExpr>(Scope))
        return Scope;
    } else if (Parent.get<FunctionDecl>() != nullptr) {
      return nullptr;
    }
    // Other nodes (types, attributes) are transparent: keep walking.
    Node = Parent;
  }
}

void RethrowCaughtExceptionCheck::registerMatchers(MatchFinder *Finder) {
  // The variable binds from the thrown operand; scope is validated per throw
  // in `check()`, where exclusions cannot suppress valid siblings.
  const auto RefToVar = ignoringParenImpCasts(
      declRefExpr(
          to(varDecl(isExceptionVariable(),
                     hasType(qualType(hasCanonicalType(referenceType()))))
                 .bind("var")))
          .bind("ref"));
  const auto MoveOfVar =
      callExpr(argumentCountIs(1), callee(functionDecl(hasName("::std::move"))),
               hasArgument(0, RefToVar));
  const auto CopyOrMoveOfVar =
      cxxConstructExpr(argumentCountIs(1),
                       hasDeclaration(cxxConstructorDecl(
                           anyOf(isCopyConstructor(), isMoveConstructor()))),
                       hasArgument(0, anyOf(RefToVar, MoveOfVar)));

  Finder->addMatcher(
      cxxThrowExpr(unless(isExpansionInSystemHeader()),
                   has(expr(anyOf(RefToVar, MoveOfVar, CopyOrMoveOfVar))))
          .bind("throw"),
      this);
}

void RethrowCaughtExceptionCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Throw = Result.Nodes.getNodeAs<CXXThrowExpr>("throw");
  const auto *Var = Result.Nodes.getNodeAs<VarDecl>("var");
  const auto *Ref = Result.Nodes.getNodeAs<DeclRefExpr>("ref");
  assert(Throw != nullptr && "throw must be bound");
  assert(Var != nullptr && "var must be bound");
  assert(Ref != nullptr && "ref must be bound");

  const SourceManager &SM = *Result.SourceManager;

  // Never touch unevaluated operands: nothing is thrown there, and a bare
  // `throw;` could change the meaning. Uses in evaluated array bounds carry
  // no such marker, so those keep warning.
  if (Ref->isNonOdrUse() == NOUR_Unevaluated)
    return;

  // The nearest scope must declare the thrown variable; otherwise a bare
  // `throw;` would rethrow the wrong exception (or terminate).
  const auto *NearestCatch =
      dyn_cast_or_null<CXXCatchStmt>(findNearestScope(Throw, Result.Context));
  if (NearestCatch == nullptr || NearestCatch->getExceptionDecl() != Var)
    return;

  const Expr *Thrown = Throw->getSubExpr();
  if (Thrown == nullptr)
    return;

  // Ignore explicitly spelled constructions (`throw Error(e);`): unlike the
  // implicit copy in `throw e;`, they deliberately build a new object.
  if (isExplicitConstruction(Thrown))
    return;

  // Defensive: never rewrite a construction of a different type. Qualifiers
  // are ignored since the thrown copy is never const-qualified.
  const QualType CaughtCanon = Var->getType()
                                   .getCanonicalType()
                                   .getNonReferenceType()
                                   .getUnqualifiedType();
  const QualType ThrownCanon = Thrown->getType()
                                   .getCanonicalType()
                                   .getNonReferenceType()
                                   .getUnqualifiedType();
  if (CaughtCanon != ThrownCanon)
    return;

  const auto Diag = diag(Throw->getThrowLoc(),
                         "throwing a copy of the caught %0 exception; use a "
                         "bare 'throw' to rethrow the original exception");
  Diag << ThrownCanon;

  // Never fix macro expansions, whose invocation sites may be shared.
  if (utils::rangeContainsMacroExpansion(Throw->getSourceRange(), &SM))
    return;
  const CharSourceRange FileRange = Lexer::makeFileCharRange(
      CharSourceRange::getTokenRange(Throw->getSourceRange()), SM,
      getLangOpts());
  if (!FileRange.isValid())
    return;
  Diag << FixItHint::CreateReplacement(FileRange, "throw");
}

} // namespace clang::tidy::bugprone
