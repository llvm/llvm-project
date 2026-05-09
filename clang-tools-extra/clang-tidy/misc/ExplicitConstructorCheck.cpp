//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ExplicitConstructorCheck.h"
#include "../utils/LexerUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::misc {

void ExplicitConstructorCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      cxxConstructorDecl(unless(anyOf(isImplicit(), // Compiler-generated.
                                      isDeleted(), isInstantiated())))
          .bind("ctor"),
      this);
  Finder->addMatcher(
      cxxConversionDecl(unless(anyOf(isExplicit(), // Already marked explicit.
                                     isImplicit(), // Compiler-generated.
                                     isDeleted(), isInstantiated())))

          .bind("conversion"),
      this);
}

static bool declIsStdInitializerList(const NamedDecl *D) {
  // First use the fast getName() method to avoid unnecessary calls to the
  // slow getQualifiedNameAsString().
  return D->getName() == "initializer_list" &&
         D->getQualifiedNameAsString() == "std::initializer_list";
}

static bool isStdInitializerList(QualType Type) {
  Type = Type.getCanonicalType();
  if (const auto *TS = Type->getAs<TemplateSpecializationType>()) {
    if (const TemplateDecl *TD = TS->getTemplateName().getAsTemplateDecl())
      return declIsStdInitializerList(TD);
  }
  if (const auto *RT = Type->getAs<RecordType>()) {
    if (const auto *Specialization =
            dyn_cast<ClassTemplateSpecializationDecl>(RT->getDecl()))
      return declIsStdInitializerList(Specialization->getSpecializedTemplate());
  }
  return false;
}

void ExplicitConstructorCheck::check(const MatchFinder::MatchResult &Result) {
  constexpr char NoExpressionWarningMessage[] =
      "%0 must be marked explicit to avoid unintentional implicit conversions";
  constexpr char WithExpressionWarningMessage[] =
      "%0 explicit expression evaluates to 'false'";

  if (const auto *Conversion =
          Result.Nodes.getNodeAs<CXXConversionDecl>("conversion")) {
    if (Conversion->isOutOfLine())
      return;
    const SourceLocation Loc = Conversion->getLocation();
    // Ignore all macros until we learn to ignore specific ones (e.g. used in
    // gmock to define matchers).
    if (Loc.isMacroID())
      return;
    diag(Loc, NoExpressionWarningMessage)
        << Conversion << FixItHint::CreateInsertion(Loc, "explicit ");
    return;
  }

  const auto *Ctor = Result.Nodes.getNodeAs<CXXConstructorDecl>("ctor");
  if (Ctor->isOutOfLine() || Ctor->getNumParams() == 0 ||
      Ctor->getMinRequiredArguments() > 1)
    return;

  const ExplicitSpecifier ExplicitSpec = Ctor->getExplicitSpecifier();

  const bool TakesInitializerList = isStdInitializerList(
      Ctor->getParamDecl(0)->getType().getNonReferenceType());
  if (ExplicitSpec.isExplicit() &&
      (Ctor->isCopyOrMoveConstructor() || TakesInitializerList)) {
    auto IsKwExplicit = [](const Token &Tok) {
      return Tok.is(tok::raw_identifier) &&
             Tok.getRawIdentifier() == "explicit";
    };
    const CharSourceRange ConstructorRange = CharSourceRange::getTokenRange(
        Ctor->getOuterLocStart(), Ctor->getEndLoc());
    const CharSourceRange ExplicitTokenRange =
        utils::lexer::findTokenTextInRange(ConstructorRange,
                                           *Result.SourceManager, getLangOpts(),
                                           IsKwExplicit);
    StringRef ConstructorDescription;
    if (Ctor->isMoveConstructor())
      ConstructorDescription = "move";
    else if (Ctor->isCopyConstructor())
      ConstructorDescription = "copy";
    else
      ConstructorDescription = "initializer-list";

    auto Diag = diag(Ctor->getLocation(),
                     "%0 constructor should not be declared explicit")
                << ConstructorDescription;
    if (ExplicitTokenRange.isValid())
      Diag << FixItHint::CreateRemoval(ExplicitTokenRange);
    return;
  }

  if (ExplicitSpec.isExplicit() || Ctor->isCopyOrMoveConstructor() ||
      TakesInitializerList)
    return;

  // Don't complain about explicit(false) or dependent expressions
  const Expr *ExplicitExpr = ExplicitSpec.getExpr();
  if (ExplicitExpr) {
    ExplicitExpr = ExplicitExpr->IgnoreImplicit();
    if (isa<CXXBoolLiteralExpr>(ExplicitExpr) ||
        ExplicitExpr->isInstantiationDependent())
      return;
  }

  const bool SingleArgument =
      Ctor->getNumParams() == 1 && !Ctor->getParamDecl(0)->isParameterPack();
  const SourceLocation Loc = Ctor->getLocation();
  auto Diag =
      diag(Loc, ExplicitExpr ? WithExpressionWarningMessage
                             : NoExpressionWarningMessage)
      << (SingleArgument
              ? "single-argument constructors"
              : "constructors that are callable with a single argument");

  if (!ExplicitExpr)
    Diag << FixItHint::CreateInsertion(Loc, "explicit ");
}

} // namespace clang::tidy::misc
