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
  auto NonConstAndNonFunctionPointerType = hasType(pointerType(unless(
      pointee(anyOf(isConstQualified(), ignoringParens(functionType()))))));

  Finder->addMatcher(
      valueDecl(
          hasType(isConstQualified()),
          hasType(typedefType(hasDeclaration(anyOf(
              typedefDecl(NonConstAndNonFunctionPointerType).bind("typedef"),
              typeAliasDecl(NonConstAndNonFunctionPointerType)
                  .bind("typeAlias"))))))
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
  ASTContext &Ctx = *Result.Context;
  QualType CanQT = Var->getType().getCanonicalType();

  SourceLocation AliasLoc;
  const char *AliasType;
  if (const auto *Typedef = Result.Nodes.getNodeAs<TypedefDecl>("typedef")) {
    AliasLoc = Typedef->getLocation();
    AliasType = "typedef";
  } else if (const auto *TypeAlias =
                 Result.Nodes.getNodeAs<TypeAliasDecl>("typeAlias")) {
    AliasLoc = TypeAlias->getLocation();
    AliasType = "type alias";
  } else {
    llvm_unreachable("registerMatchers has registered an unknown matcher,"
                     " code out of sync");
  }

  diag(Var->getLocation(), "%0 declared with a const-qualified %1; "
                           "results in the type being '%2' instead of '%3'")
      << Var << AliasType << CanQT.getAsString(Ctx.getPrintingPolicy())
      << guessAlternateQualification(Ctx, CanQT)
             .getAsString(Ctx.getPrintingPolicy());
  diag(AliasLoc, "%0 declared here", DiagnosticIDs::Note) << AliasType;
}

} // namespace misc
} // namespace tidy
} // namespace clang
