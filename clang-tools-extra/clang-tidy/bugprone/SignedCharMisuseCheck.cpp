//===--- SignedCharMisuseCheck.cpp - clang-tidy ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SignedCharMisuseCheck.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;
using namespace clang::ast_matchers::internal;

namespace clang {
namespace tidy {
namespace bugprone {

static Matcher<TypedefDecl> hasAnyListedName(const std::string &Names) {
  const std::vector<std::string> NameList =
      utils::options::parseStringList(Names);
  return hasAnyName(std::vector<StringRef>(NameList.begin(), NameList.end()));
}

SignedCharMisuseCheck::SignedCharMisuseCheck(StringRef Name,
                                             ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      CharTypdefsToIgnoreList(Options.get("CharTypdefsToIgnore", "")) {}

void SignedCharMisuseCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "CharTypdefsToIgnore", CharTypdefsToIgnoreList);
}

void SignedCharMisuseCheck::registerMatchers(MatchFinder *Finder) {
  // We can ignore typedefs which are some kind of integer types
  // (e.g. typedef char sal_Int8). In this case, we don't need to
  // worry about the misinterpretation of char values.
  const auto IntTypedef = qualType(
      hasDeclaration(typedefDecl(hasAnyListedName(CharTypdefsToIgnoreList))));

  const auto SignedCharType = expr(hasType(qualType(
      allOf(isAnyCharacter(), isSignedInteger(), unless(IntTypedef)))));

  const auto IntegerType = qualType(allOf(isInteger(), unless(isAnyCharacter()),
                                          unless(booleanType())))
                               .bind("integerType");

  // We are interested in signed char -> integer conversion.
  const auto ImplicitCastExpr =
      implicitCastExpr(hasSourceExpression(SignedCharType),
                       hasImplicitDestinationType(IntegerType))
          .bind("castExpression");

  const auto CStyleCastExpr = cStyleCastExpr(has(ImplicitCastExpr));
  const auto StaticCastExpr = cxxStaticCastExpr(has(ImplicitCastExpr));
  const auto FunctionalCastExpr = cxxFunctionalCastExpr(has(ImplicitCastExpr));

  // We catch any type of casts to an integer. We need to have these cast
  // expressions explicitly to catch only those casts which are direct children
  // of an assignment/declaration.
  const auto CastExpr = expr(anyOf(ImplicitCastExpr, CStyleCastExpr,
                                   StaticCastExpr, FunctionalCastExpr));

  // Catch assignments with the suspicious type conversion.
  const auto AssignmentOperatorExpr = expr(binaryOperator(
      hasOperatorName("="), hasLHS(hasType(IntegerType)), hasRHS(CastExpr)));

  Finder->addMatcher(AssignmentOperatorExpr, this);

  // Catch declarations with the suspicious type conversion.
  const auto Declaration =
      varDecl(isDefinition(), hasType(IntegerType), hasInitializer(CastExpr));

  Finder->addMatcher(Declaration, this);
}

void SignedCharMisuseCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *CastExpression =
      Result.Nodes.getNodeAs<ImplicitCastExpr>("castExpression");
  const auto *IntegerType = Result.Nodes.getNodeAs<QualType>("integerType");
  assert(CastExpression);
  assert(IntegerType);

  // Ignore the match if we know that the value is not negative.
  // The potential misinterpretation happens for negative values only.
  Expr::EvalResult EVResult;
  if (!CastExpression->isValueDependent() &&
      CastExpression->getSubExpr()->EvaluateAsInt(EVResult, *Result.Context)) {
    llvm::APSInt Value1 = EVResult.Val.getInt();
    if (Value1.isNonNegative())
      return;
  }

  diag(CastExpression->getBeginLoc(),
       "'signed char' to %0 conversion; "
       "consider casting to 'unsigned char' first.")
      << *IntegerType;
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
