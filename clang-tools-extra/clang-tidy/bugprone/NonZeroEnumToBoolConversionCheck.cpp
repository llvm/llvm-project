//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NonZeroEnumToBoolConversionCheck.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include <algorithm>

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

namespace {

AST_MATCHER(EnumDecl, isCompleteAndHasNoZeroValue) {
  const EnumDecl *Definition = Node.getDefinition();
  return Definition && Node.isComplete() &&
         llvm::none_of(Definition->enumerators(),
                       [](const EnumConstantDecl *Value) {
                         return Value->getInitVal().isZero();
                       });
}

} // namespace

NonZeroEnumToBoolConversionCheck::NonZeroEnumToBoolConversionCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      EnumIgnoreList(
          utils::options::parseStringList(Options.get("EnumIgnoreList", ""))) {}

void NonZeroEnumToBoolConversionCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "EnumIgnoreList",
                utils::options::serializeStringList(EnumIgnoreList));
}

bool NonZeroEnumToBoolConversionCheck::isLanguageVersionSupported(
    const LangOptions &LangOpts) const {
  return LangOpts.CPlusPlus;
}

void NonZeroEnumToBoolConversionCheck::registerMatchers(MatchFinder *Finder) {
  // Excluding bitwise operators (binary and overload) to avoid false-positives
  // in code like this 'if (e & SUCCESS) {'.
  auto ExcludedOperators = binaryOperation(hasAnyOperatorName(
      "|", "&", "^", "<<", ">>", "~", "|=", "&=", "^=", "<<=", ">>="));

  Finder->addMatcher(
      castExpr(hasCastKind(CK_IntegralToBoolean),
               unless(isExpansionInSystemHeader()), hasType(booleanType()),
               hasSourceExpression(
                   expr(hasType(qualType(hasCanonicalType(hasDeclaration(
                            enumDecl(isCompleteAndHasNoZeroValue(),
                                     unless(matchers::matchesAnyListedName(
                                         EnumIgnoreList)))
                                .bind("enum"))))),
                        unless(declRefExpr(to(enumConstantDecl()))),
                        unless(ignoringParenImpCasts(ExcludedOperators)))),
               unless(hasAncestor(staticAssertDecl())))
          .bind("cast"),
      this);
}

void NonZeroEnumToBoolConversionCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Cast = Result.Nodes.getNodeAs<CastExpr>("cast");
  const auto *Enum = Result.Nodes.getNodeAs<EnumDecl>("enum");

  diag(Cast->getExprLoc(), "conversion of %0 into 'bool' will always return "
                           "'true', enum doesn't have a zero-value enumerator")
      << Enum;
  diag(Enum->getLocation(), "enum is defined here", DiagnosticIDs::Note);
}

} // namespace clang::tidy::bugprone
