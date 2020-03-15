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

static constexpr int UnsignedASCIIUpperBound = 127;

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

// Create a matcher for char -> integer cast.
BindableMatcher<clang::Stmt> SignedCharMisuseCheck::charCastExpression(
    bool IsSigned, const Matcher<clang::QualType> &IntegerType,
    const std::string &CastBindName) const {
  // We can ignore typedefs which are some kind of integer types
  // (e.g. typedef char sal_Int8). In this case, we don't need to
  // worry about the misinterpretation of char values.
  const auto IntTypedef = qualType(
      hasDeclaration(typedefDecl(hasAnyListedName(CharTypdefsToIgnoreList))));

  auto CharTypeExpr = expr();
  if (IsSigned) {
    CharTypeExpr = expr(hasType(
        qualType(isAnyCharacter(), isSignedInteger(), unless(IntTypedef))));
  } else {
    CharTypeExpr = expr(hasType(qualType(
        isAnyCharacter(), unless(isSignedInteger()), unless(IntTypedef))));
  }

  const auto ImplicitCastExpr =
      implicitCastExpr(hasSourceExpression(CharTypeExpr),
                       hasImplicitDestinationType(IntegerType))
          .bind(CastBindName);

  const auto CStyleCastExpr = cStyleCastExpr(has(ImplicitCastExpr));
  const auto StaticCastExpr = cxxStaticCastExpr(has(ImplicitCastExpr));
  const auto FunctionalCastExpr = cxxFunctionalCastExpr(has(ImplicitCastExpr));

  // We catch any type of casts to an integer. We need to have these cast
  // expressions explicitly to catch only those casts which are direct children
  // of the checked expressions. (e.g. assignment, declaration).
  return expr(anyOf(ImplicitCastExpr, CStyleCastExpr, StaticCastExpr,
                    FunctionalCastExpr));
}

void SignedCharMisuseCheck::registerMatchers(MatchFinder *Finder) {
  const auto IntegerType =
      qualType(isInteger(), unless(isAnyCharacter()), unless(booleanType()))
          .bind("integerType");
  const auto SignedCharCastExpr =
      charCastExpression(true, IntegerType, "signedCastExpression");
  const auto UnSignedCharCastExpr =
      charCastExpression(false, IntegerType, "unsignedCastExpression");

  // Catch assignments with singed char -> integer conversion.
  const auto AssignmentOperatorExpr =
      expr(binaryOperator(hasOperatorName("="), hasLHS(hasType(IntegerType)),
                          hasRHS(SignedCharCastExpr)));

  Finder->addMatcher(AssignmentOperatorExpr, this);

  // Catch declarations with singed char -> integer conversion.
  const auto Declaration = varDecl(isDefinition(), hasType(IntegerType),
                                   hasInitializer(SignedCharCastExpr));

  Finder->addMatcher(Declaration, this);

  // Catch signed char/unsigned char comparison.
  const auto CompareOperator =
      expr(binaryOperator(hasAnyOperatorName("==", "!="),
                          anyOf(allOf(hasLHS(SignedCharCastExpr),
                                      hasRHS(UnSignedCharCastExpr)),
                                allOf(hasLHS(UnSignedCharCastExpr),
                                      hasRHS(SignedCharCastExpr)))))
          .bind("comparison");

  Finder->addMatcher(CompareOperator, this);
}

void SignedCharMisuseCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *SignedCastExpression =
      Result.Nodes.getNodeAs<ImplicitCastExpr>("signedCastExpression");

  // Ignore the match if we know that the signed char's value is not negative.
  // The potential misinterpretation happens for negative values only.
  Expr::EvalResult EVResult;
  if (!SignedCastExpression->isValueDependent() &&
      SignedCastExpression->getSubExpr()->EvaluateAsInt(EVResult,
                                                        *Result.Context)) {
    llvm::APSInt Value = EVResult.Val.getInt();
    if (Value.isNonNegative())
      return;
  }

  if (const auto *Comparison = Result.Nodes.getNodeAs<Expr>("comparison")) {
    const auto *UnSignedCastExpression =
        Result.Nodes.getNodeAs<ImplicitCastExpr>("unsignedCastExpression");

    // We can ignore the ASCII value range also for unsigned char.
    Expr::EvalResult EVResult;
    if (!UnSignedCastExpression->isValueDependent() &&
        UnSignedCastExpression->getSubExpr()->EvaluateAsInt(EVResult,
                                                            *Result.Context)) {
      llvm::APSInt Value = EVResult.Val.getInt();
      if (Value <= UnsignedASCIIUpperBound)
        return;
    }

    diag(Comparison->getBeginLoc(),
         "comparison between 'signed char' and 'unsigned char'");
  } else if (const auto *IntegerType =
                 Result.Nodes.getNodeAs<QualType>("integerType")) {
    diag(SignedCastExpression->getBeginLoc(),
         "'signed char' to %0 conversion; "
         "consider casting to 'unsigned char' first.")
        << *IntegerType;
  } else
    llvm_unreachable("Unexpected match");
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
