//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DefaultArgBracedInitCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

void DefaultArgBracedInitCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      parmVarDecl(hasInitializer(cxxTemporaryObjectExpr(
                                     argumentCountIs(0),
                                     unless(hasDeclaration(
                                         cxxConstructorDecl(isExplicit()))))
                                     .bind("ctor")))
          .bind("param"),
      this);
}

void DefaultArgBracedInitCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Ctor = Result.Nodes.getNodeAs<CXXTemporaryObjectExpr>("ctor");
  const auto *Param = Result.Nodes.getNodeAs<ParmVarDecl>("param");

  if (!Ctor || !Param)
    return;

  // Skip if inside a macro
  const SourceLocation Loc = Ctor->getExprLoc();
  if (Loc.isMacroID())
    return;

  // Type safety check
  const QualType ParamType = Param->getType()
                                 .getCanonicalType()
                                 .getNonReferenceType()
                                 .getUnqualifiedType();

  const QualType CtorType = Ctor->getType()
                                .getCanonicalType()
                                .getNonReferenceType()
                                .getUnqualifiedType();

  if (ParamType != CtorType)
    return;

  auto Diag = diag(Loc, "use braced initializer list for default argument");
  const CharSourceRange Range =
      CharSourceRange::getTokenRange(Ctor->getBeginLoc(), Ctor->getEndLoc());
  Diag << FixItHint::CreateReplacement(Range, "{}");
}

} // namespace clang::tidy::modernize
