//===--- UseStdNumbersCheck.cpp - clang_tidy ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX_License_Identifier: Apache_2.0 WITH LLVM_exception
//
//===----------------------------------------------------------------------===//

#include "UseStdNumbersCheck.h"
#include "../ClangTidyDiagnosticConsumer.h"
#include "../utils/TransformerClangTidyCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchersInternal.h"
#include "clang/ASTMatchers/ASTMatchersMacros.h"
#include "clang/Basic/LLVM.h"
#include "clang/Tooling/Transformer/RewriteRule.h"
#include "clang/Tooling/Transformer/Stencil.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MathExtras.h"
#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>

namespace {
using namespace clang::ast_matchers;
using clang::ast_matchers::internal::Matcher;
using clang::transformer::addInclude;
using clang::transformer::applyFirst;
using clang::transformer::ASTEdit;
using clang::transformer::cat;
using clang::transformer::changeTo;
using clang::transformer::edit;
using clang::transformer::EditGenerator;
using clang::transformer::flattenVector;
using clang::transformer::RewriteRuleWith;
using llvm::StringRef;

constexpr auto DiffThreshold = 0.001;

AST_MATCHER_P(clang::FloatingLiteral, near, double, Value) {
  return std::abs(Node.getValueAsApproximateDouble() - Value) < DiffThreshold;
}

AST_MATCHER_P(clang::QualType, hasCanonicalTypeUnqualified,
              Matcher<clang::QualType>, InnerMatcher) {
  return !Node.isNull() &&
         InnerMatcher.matches(Node->getCanonicalTypeUnqualified(), Finder,
                              Builder);
}

AST_MATCHER(clang::QualType, isArithmetic) {
  return !Node.isNull() && Node->isArithmeticType();
}
AST_MATCHER(clang::QualType, isFloating) {
  return !Node.isNull() && Node->isFloatingType();
}

auto ignoreImplicitAndArithmeticCasting(const Matcher<clang::Expr> Matcher) {
  return expr(
      ignoringImplicit(expr(hasType(qualType(isArithmetic())),
                            ignoringParenCasts(ignoringImplicit(Matcher)))));
}

auto ignoreImplicitAndFloatingCasting(const Matcher<clang::Expr> Matcher) {
  return expr(
      ignoringImplicit(expr(hasType(qualType(isFloating())),
                            ignoringParenCasts(ignoringImplicit(Matcher)))));
}

auto matchMathCall(const StringRef FunctionName,
                   const Matcher<clang::Expr> ArgumentMatcher) {
  return callExpr(
      callee(functionDecl(hasName(FunctionName),
                          hasParameter(0, hasType(isArithmetic())))),
      hasArgument(0, ArgumentMatcher));
}

auto matchSqrt(const Matcher<clang::Expr> ArgumentMatcher) {
  return matchMathCall("sqrt", ArgumentMatcher);
}

// Used for top-level matchers (i.e. the match that replaces Val with its
// constant).
//
// E.g. The matcher of `std::numbers::pi` uses this matcher to look to
// floatLiterals that have the value of pi.
//
// We only care about the literal if the match is for a top-level match
auto matchFloatLiteralNear(const double Val) {
  return expr(ignoreImplicitAndFloatingCasting(floatLiteral(near(Val))));
}

// Used for non-top-level matchers (i.e. matchers that are used as inner
// matchers for top-level matchers).
//
// E.g.: The matcher of `std::numbers::log2e` uses this matcher to check if `e`
// of `log2(e)` is declared constant and initialized with the value for eulers
// number.
//
// Here, we do care about literals and about DeclRefExprs to variable
// declarations that are constant and initialized with `Val`. This allows
// top-level matchers to see through declared constants for their inner matches
// like the `std::numbers::log2e` matcher.
auto matchFloatValueNear(const double Val) {
  const auto Float = floatLiteral(near(Val));

  const auto Dref = declRefExpr(
      to(varDecl(hasType(qualType(isConstQualified(), isFloating())),
                 hasInitializer(ignoreImplicitAndFloatingCasting(Float)))));
  return expr(ignoreImplicitAndFloatingCasting(anyOf(Float, Dref)));
}

auto matchValue(const int64_t ValInt) {
  const auto Int =
      expr(ignoreImplicitAndArithmeticCasting(integerLiteral(equals(ValInt))));
  const auto Float = expr(ignoreImplicitAndFloatingCasting(
      matchFloatValueNear(static_cast<double>(ValInt))));
  const auto Dref = declRefExpr(to(varDecl(
      anyOf(isConstexpr(),
            varDecl(hasType(qualType(isConstQualified(), isArithmetic())))),
      hasInitializer(anyOf(Int, Float)))));
  return expr(anyOf(Int, Float, Dref));
}

auto match1Div(const Matcher<clang::Expr> Match) {
  return binaryOperator(hasOperatorName("/"), hasLHS(matchValue(1)),
                        hasRHS(ignoringImplicit(Match)));
}

auto matchEuler() {
  return expr(anyOf(matchFloatValueNear(llvm::numbers::e),
                    matchMathCall("exp", matchValue(1))));
}
auto matchEulerTopLevel() {
  return expr(anyOf(matchFloatLiteralNear(llvm::numbers::e),
                    matchMathCall("exp", matchValue(1))));
}

auto matchLog2Euler() {
  return expr(anyOf(matchFloatLiteralNear(llvm::numbers::log2e),
                    matchMathCall("log2", matchEuler())));
}

auto matchLog10Euler() {
  return expr(anyOf(matchFloatLiteralNear(llvm::numbers::log10e),
                    matchMathCall("log10", matchEuler())));
}

auto matchPi() { return matchFloatValueNear(llvm::numbers::pi); }
auto matchPiTopLevel() { return matchFloatLiteralNear(llvm::numbers::pi); }

auto matchEgamma() { return matchFloatLiteralNear(llvm::numbers::egamma); }

auto matchInvPi() {
  return expr(anyOf(matchFloatLiteralNear(llvm::numbers::inv_pi),
                    match1Div(matchPi())));
}

auto matchInvSqrtPi() {
  return expr(anyOf(matchFloatLiteralNear(llvm::numbers::inv_sqrtpi),
                    match1Div(matchSqrt(matchPi()))));
}

auto matchLn2() {
  return expr(anyOf(matchFloatLiteralNear(llvm::numbers::ln2),
                    matchMathCall("log", ignoringImplicit(matchValue(2)))));
}

auto machterLn10() {
  return expr(anyOf(matchFloatLiteralNear(llvm::numbers::ln10),
                    matchMathCall("log", ignoringImplicit(matchValue(10)))));
}

auto matchSqrt2() {
  return expr(anyOf(matchFloatLiteralNear(llvm::numbers::sqrt2),
                    matchSqrt(matchValue(2))));
}

auto matchSqrt3() {
  return expr(anyOf(matchFloatLiteralNear(llvm::numbers::sqrt3),
                    matchSqrt(matchValue(3))));
}

auto matchInvSqrt3() {
  return expr(anyOf(matchFloatLiteralNear(llvm::numbers::inv_sqrt3),
                    match1Div(matchSqrt(matchValue(3)))));
}

auto matchPhi() {
  const auto PhiFormula = binaryOperator(
      hasOperatorName("/"),
      hasLHS(parenExpr(has(binaryOperator(
          hasOperatorName("+"), hasEitherOperand(matchValue(1)),
          hasEitherOperand(matchMathCall("sqrt", matchValue(5))))))),
      hasRHS(matchValue(2)));
  return expr(anyOf(PhiFormula, matchFloatLiteralNear(llvm::numbers::phi)));
}

EditGenerator applyRuleForBoundOrDefault(
    ASTEdit DefaultEdit,
    llvm::SmallVector<std::pair<StringRef, ASTEdit>, 2> Edits) {
  return [Edits = std::move(Edits), DefaultEdit = std::move(DefaultEdit)](
             const MatchFinder::MatchResult &Result) {
    auto &Map = Result.Nodes.getMap();
    for (const auto &[Id, EditOfId] : Edits) {
      if (Map.find(Id) != Map.end()) {
        return edit(EditOfId)(Result);
      }
    }
    return edit(DefaultEdit)(Result);
  };
}

RewriteRuleWith<std::string> makeRule(const Matcher<clang::Stmt> Matcher,
                                      const StringRef Constant) {
  static const auto AddNumbersInclude =
      addInclude("numbers", clang::transformer::IncludeFormat::Angled);

  const auto DefaultEdit = changeTo(cat("std::numbers::", Constant));
  const auto FloatEdit = changeTo(cat("std::numbers::", Constant, "_v<float>"));
  const auto LongDoubleEdit =
      changeTo(cat("std::numbers::", Constant, "_v<long double>"));

  const auto EditRules = applyRuleForBoundOrDefault(
      DefaultEdit, {{"float", FloatEdit}, {"long double", LongDoubleEdit}});

  return makeRule(
      expr(ignoreImplicitAndFloatingCasting(Matcher),
           unless(isInTemplateInstantiation()),
           hasType(qualType(hasCanonicalType(hasCanonicalTypeUnqualified(anyOf(
               qualType(asString("float")).bind("float"),
               qualType(asString("double")),
               qualType(asString("long double")).bind("long double"))))))),
      flattenVector({edit(AddNumbersInclude), EditRules}),
      cat("prefer std::numbers math constant"));
}

/*
  List of all math constants
  + e
  + log2e
  + log10e
  + pi
  + inv_pi
  + inv_sqrtpi
  + ln2
  + ln10
  + sqrt2
  + sqrt3
  + inv_sqrt3
  + egamma
  + phi
*/

RewriteRuleWith<std::string> makeRewriteRule() {
  return applyFirst({
      makeRule(matchLog2Euler(), "log2e"),
      makeRule(matchLog10Euler(), "log10e"),
      makeRule(matchEulerTopLevel(), "e"),
      makeRule(matchEgamma(), "egamma"),
      makeRule(matchInvSqrtPi(), "inv_sqrtpi"),
      makeRule(matchInvPi(), "inv_pi"),
      makeRule(matchPiTopLevel(), "pi"),
      makeRule(matchLn2(), "ln2"),
      makeRule(machterLn10(), "ln10"),
      makeRule(matchSqrt2(), "sqrt2"),
      makeRule(matchInvSqrt3(), "inv_sqrt3"),
      makeRule(matchSqrt3(), "sqrt3"),
      makeRule(matchPhi(), "phi"),
  });
}
} // namespace

namespace clang::tidy::modernize {
UseStdNumbersCheck::UseStdNumbersCheck(const StringRef Name,
                                       ClangTidyContext *const Context)
    : TransformerClangTidyCheck(Name, Context) {
  setRule(makeRewriteRule());
}
} // namespace clang::tidy::modernize
