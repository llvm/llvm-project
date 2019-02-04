//===--- MisplacedConstCheck.cpp - clang-tidy------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MisplacedConstCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace misc {

void MisplacedConstCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      valueDecl(hasType(isConstQualified()),
                hasType(typedefType(hasDeclaration(
                    typedefDecl(hasType(pointerType(unless(pointee(
                                    anyOf(isConstQualified(),
                                          ignoringParens(functionType())))))))
                        .bind("typedef")))))
          .bind("decl"),
      this);
}

static QualType guessAlternateQualification(ASTContext &Context, QualType QT) {
  // We're given a QualType from a typedef where the qualifiers apply to the
  // pointer instead of the pointee. Strip the const qualifier from the pointer
  // type and add it to the pointee instead.
  if (!QT->isPointerType())
    return QT;

  Qualifiers Quals = QT.getLocalQualifiers();
  Quals.removeConst();

  QualType NewQT = Context.getPointerType(
      QualType(QT->getPointeeType().getTypePtr(), Qualifiers::Const));
  return NewQT.withCVRQualifiers(Quals.getCVRQualifiers());
}

void MisplacedConstCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Var = Result.Nodes.getNodeAs<ValueDecl>("decl");
  const auto *Typedef = Result.Nodes.getNodeAs<TypedefDecl>("typedef");
  ASTContext &Ctx = *Result.Context;
  QualType CanQT = Var->getType().getCanonicalType();

  diag(Var->getLocation(), "%0 declared with a const-qualified typedef type; "
                           "results in the type being '%1' instead of '%2'")
      << Var << CanQT.getAsString(Ctx.getPrintingPolicy())
      << guessAlternateQualification(Ctx, CanQT)
             .getAsString(Ctx.getPrintingPolicy());
  diag(Typedef->getLocation(), "typedef declared here", DiagnosticIDs::Note);
}

} // namespace misc
} // namespace tidy
} // namespace clang
