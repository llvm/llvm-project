//===-- UseAuto/UseAutoActions.cpp - Matcher callback impl ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
///  \file
///  \brief This file contains the implementation of callbacks for the UseAuto
///  transform.
///
//===----------------------------------------------------------------------===//
#include "UseAutoActions.h"
#include "UseAutoMatchers.h"
#include "clang/AST/ASTContext.h"

using namespace clang::ast_matchers;
using namespace clang::tooling;
using namespace clang;

void IteratorReplacer::run(const MatchFinder::MatchResult &Result) {
  const VarDecl *D = Result.Nodes.getNodeAs<VarDecl>(IteratorDeclId);

  assert(D && "Bad Callback. No node provided");

  SourceManager &SM = *Result.SourceManager;
  if (!SM.isFromMainFile(D->getLocStart()))
    return;

  const Expr *ExprInit = D->getInit();

  // Skip expressions with cleanups from the initializer expression.
  if (const ExprWithCleanups *E = dyn_cast<ExprWithCleanups>(ExprInit))
    ExprInit = E->getSubExpr();

  const CXXConstructExpr *Construct = cast<CXXConstructExpr>(ExprInit);

  assert(Construct->getNumArgs() == 1u &&
         "Expected constructor with single argument");

  // Drill down to the as-written initializer.
  const Expr *E = Construct->arg_begin()->IgnoreParenImpCasts();
  if (E != E->IgnoreConversionOperator())
    // We hit a conversion operator. Early-out now as they imply an implicit
    // conversion from a different type. Could also mean an explicit conversion
    // from the same type but that's pretty rare.
    return;

  if (const CXXConstructExpr *NestedConstruct = dyn_cast<CXXConstructExpr>(E))
    // If we ran into an implicit conversion constructor, can't convert.
    //
    // FIXME: The following only checks if the constructor can be used
    // implicitly, not if it actually was. Cases where the converting constructor
    // was used explicitly won't get converted.
    if (NestedConstruct->getConstructor()->isConvertingConstructor(false))
      return;

  if (Result.Context->hasSameType(D->getType(), E->getType())) {
    TypeLoc TL = D->getTypeSourceInfo()->getTypeLoc();

    // WARNING: TypeLoc::getSourceRange() will include the identifier for things
    // like function pointers. Not a concern since this action only works with
    // iterators but something to keep in mind in the future.

    CharSourceRange Range(TL.getSourceRange(), true);
    Replace.insert(tooling::Replacement(SM, Range, "auto"));
    ++AcceptedChanges;
  }
}

void NewReplacer::run(const MatchFinder::MatchResult &Result) {
  const VarDecl *D = Result.Nodes.getNodeAs<VarDecl>(DeclWithNewId);
  assert(D && "Bad Callback. No node provided");

  SourceManager &SM = *Result.SourceManager;
  if (!SM.isFromMainFile(D->getLocStart()))
    return;
  
  const CXXNewExpr *NewExpr = Result.Nodes.getNodeAs<CXXNewExpr>(NewExprId);
  assert(NewExpr && "Bad Callback. No CXXNewExpr bound");

  // If declaration and initializer have exactly the same type, just replace
  // with 'auto'.
  if (Result.Context->hasSameType(D->getType(), NewExpr->getType())) {
    TypeLoc TL = D->getTypeSourceInfo()->getTypeLoc();
    CharSourceRange Range(TL.getSourceRange(), /*IsTokenRange=*/ true);
    // Space after 'auto' to handle cases where the '*' in the pointer type
    // is next to the identifier. This avoids changing 'int *p' into 'autop'.
    Replace.insert(tooling::Replacement(SM, Range, "auto "));
    ++AcceptedChanges;
    return;
  }

  // If the CV qualifiers for the pointer differ then we still use auto, just
  // need to leave the qualifier behind.
  if (Result.Context->hasSameUnqualifiedType(D->getType(),
                                             NewExpr->getType())) {
    TypeLoc TL = D->getTypeSourceInfo()->getTypeLoc();
    CharSourceRange Range(TL.getSourceRange(), /*IsTokenRange=*/ true);
    // Space after 'auto' to handle cases where the '*' in the pointer type
    // is next to the identifier. This avoids changing 'int *p' into 'autop'.
    Replace.insert(tooling::Replacement(SM, Range, "auto "));
    ++AcceptedChanges;
    return;
  }

  // The VarDecl and Initializer have mismatching types.
  return;

  // FIXME: There is, however, one case we can address: when the VarDecl
  // pointee is the same as the initializer, just more CV-qualified. However,
  // TypeLoc information is not reliable where CV qualifiers are concerned so
  // we can't do anything about this case for now.
}
