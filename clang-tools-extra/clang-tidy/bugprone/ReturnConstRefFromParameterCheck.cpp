//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ReturnConstRefFromParameterCheck.h"
#include "clang/AST/Attrs.inc"
#include "clang/AST/Expr.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

namespace {

AST_MATCHER(ParmVarDecl, hasLifetimeBoundAttr) {
  return Node.hasAttr<LifetimeBoundAttr>();
}

} // namespace

void ReturnConstRefFromParameterCheck::registerMatchers(MatchFinder *Finder) {
  const auto DRef = ignoringParens(
      declRefExpr(
          to(parmVarDecl(hasType(hasCanonicalType(
                             qualType(lValueReferenceType(pointee(
                                          qualType(isConstQualified()))))
                                 .bind("type"))),
                         hasDeclContext(functionDecl(
                             equalsBoundNode("func"),
                             hasReturnTypeLoc(loc(qualType(
                                 hasCanonicalType(equalsBoundNode("type"))))))),
                         unless(hasLifetimeBoundAttr()))
                 .bind("param")))
          .bind("dref"));

  Finder->addMatcher(
      returnStmt(
          hasAncestor(functionDecl().bind("func")),
          hasReturnValue(anyOf(
              DRef, ignoringParens(conditionalOperator(eachOf(
                        hasTrueExpression(DRef), hasFalseExpression(DRef))))))),
      this);
}

static bool isSameTypeIgnoringConst(QualType A, QualType B) {
  return A.getCanonicalType().withConst() == B.getCanonicalType().withConst();
}

static bool isSameTypeIgnoringConstRef(QualType A, QualType B) {
  return isSameTypeIgnoringConst(A.getCanonicalType().getNonReferenceType(),
                                 B.getCanonicalType().getNonReferenceType());
}

static bool hasSameParameterTypes(const FunctionDecl &FD, const FunctionDecl &O,
                                  const ParmVarDecl &PD) {
  if (FD.getNumParams() != O.getNumParams())
    return false;
  for (unsigned I = 0, E = FD.getNumParams(); I < E; ++I) {
    const ParmVarDecl *DPD = FD.getParamDecl(I);
    const QualType OPT = O.getParamDecl(I)->getType();
    if (DPD == &PD) {
      if (!llvm::isa<RValueReferenceType>(OPT) ||
          !isSameTypeIgnoringConstRef(DPD->getType(), OPT))
        return false;
    } else {
      if (!isSameTypeIgnoringConst(DPD->getType(), OPT))
        return false;
    }
  }
  return true;
}

static const Decl *findRVRefOverload(const FunctionDecl &FD,
                                     const ParmVarDecl &PD) {
  // Actually it would be better to do lookup in caller site.
  // But in most of cases, overloads of LVRef and RVRef will appear together.
  // FIXME:
  // 1. overload in anonymous namespace
  // 2. forward reference
  DeclContext::lookup_result LookupResult =
      FD.getParent()->lookup(FD.getNameInfo().getName());
  if (LookupResult.isSingleResult()) {
    return nullptr;
  }
  for (const Decl *Overload : LookupResult) {
    if (Overload == &FD)
      continue;
    if (const auto *O = dyn_cast<FunctionDecl>(Overload))
      if (hasSameParameterTypes(FD, *O, PD))
        return O;
  }
  return nullptr;
}

void ReturnConstRefFromParameterCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *FD = Result.Nodes.getNodeAs<FunctionDecl>("func");
  const auto *PD = Result.Nodes.getNodeAs<ParmVarDecl>("param");
  const auto *DRef = Result.Nodes.getNodeAs<DeclRefExpr>("dref");
  const SourceRange Range = DRef->getSourceRange();
  if (Range.isInvalid())
    return;

  if (findRVRefOverload(*FD, *PD) != nullptr)
    return;

  diag(Range.getBegin(),
       "returning a constant reference parameter may cause use-after-free "
       "when the parameter is constructed from a temporary")
      << Range;
}

} // namespace clang::tidy::bugprone
