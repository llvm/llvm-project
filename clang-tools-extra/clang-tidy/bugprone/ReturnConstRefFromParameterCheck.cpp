//===--- ReturnConstRefFromParameterCheck.cpp - clang-tidy ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ReturnConstRefFromParameterCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

void ReturnConstRefFromParameterCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      returnStmt(
          hasReturnValue(declRefExpr(
              to(parmVarDecl(hasType(hasCanonicalType(
                                 qualType(lValueReferenceType(pointee(
                                              qualType(isConstQualified()))))
                                     .bind("type"))))
                     .bind("param")))),
          hasAncestor(
              functionDecl(hasReturnTypeLoc(loc(qualType(
                               hasCanonicalType(equalsBoundNode("type"))))))
                  .bind("func")))
          .bind("ret"),
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
  const auto *R = Result.Nodes.getNodeAs<ReturnStmt>("ret");
  const SourceRange Range = R->getRetValue()->getSourceRange();
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
