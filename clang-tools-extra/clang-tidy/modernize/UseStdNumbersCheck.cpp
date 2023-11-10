//===--- UseStdNumbersCheck.cpp - clang_tidy ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX_License_Identifier: Apache_2.0 WITH LLVM_exception
//
//===----------------------------------------------------------------------===//

#include "UseStdNumbersCheck.h"
#include "../ClangTidyDiagnosticConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchersInternal.h"
#include "clang/ASTMatchers/ASTMatchersMacros.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MathExtras.h"
#include <array>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <string>
#include <tuple>
#include <utility>

namespace {
using namespace clang::ast_matchers;
using clang::ast_matchers::internal::Matcher;
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

AST_MATCHER_P(clang::Expr, anyOfExhaustive,
              llvm::ArrayRef<Matcher<clang::Stmt>>, Exprs) {
  bool FoundMatch = false;
  for (const auto &InnerMatcher : Exprs) {
    clang::ast_matchers::internal::BoundNodesTreeBuilder Result = *Builder;
    if (InnerMatcher.matches(Node, Finder, &Result)) {
      *Builder = std::move(Result);
      FoundMatch = true;
    }
  }
  return FoundMatch;
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
  return expr(ignoreImplicitAndFloatingCasting(
      callExpr(callee(functionDecl(hasName(FunctionName),
                                   hasParameter(0, hasType(isArithmetic())))),
               hasArgument(0, ArgumentMatcher))));
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
// If the match is for a top-level match, we only care about the literal.
auto matchFloatLiteralNear(const StringRef Constant, const double Val) {
  return expr(
      ignoreImplicitAndFloatingCasting(floatLiteral(near(Val)).bind(Constant)));
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
      hasType(qualType(isConstQualified(), isArithmetic())),
      hasInitializer(expr(anyOf(ignoringImplicit(Int),
                                ignoreImplicitAndFloatingCasting(Float)))))));
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
  return expr(anyOf(matchFloatLiteralNear("e_literal", llvm::numbers::e),
                    matchMathCall("exp", matchValue(1)).bind("e_pattern")))
      .bind("e");
}

auto matchLog2Euler() {
  return expr(
             anyOf(matchFloatLiteralNear("log2e_literal", llvm::numbers::log2e),
                   matchMathCall("log2", matchEuler()).bind("log2e_pattern")))
      .bind("log2e");
}

auto matchLog10Euler() {
  return expr(
             anyOf(
                 matchFloatLiteralNear("log10e_literal", llvm::numbers::log10e),
                 matchMathCall("log10", matchEuler()).bind("log10e_pattern")))
      .bind("log10e");
}

auto matchPi() { return matchFloatValueNear(llvm::numbers::pi); }
auto matchPiTopLevel() {
  return matchFloatLiteralNear("pi_literal", llvm::numbers::pi).bind("pi");
}

auto matchEgamma() {
  return matchFloatLiteralNear("egamma_literal", llvm::numbers::egamma)
      .bind("egamma");
}

auto matchInvPi() {
  return expr(anyOf(matchFloatLiteralNear("inv_pi_literal",
                                          llvm::numbers::inv_pi),
                    match1Div(matchPi()).bind("inv_pi_pattern")))
      .bind("inv_pi");
}

auto matchInvSqrtPi() {
  return expr(anyOf(matchFloatLiteralNear("inv_sqrtpi_literal",
                                          llvm::numbers::inv_sqrtpi),
                    match1Div(matchSqrt(matchPi())).bind("inv_sqrtpi_pattern")))
      .bind("inv_sqrtpi");
}

auto matchLn2() {
  return expr(anyOf(matchFloatLiteralNear("ln2_literal", llvm::numbers::ln2),
                    matchMathCall("log", ignoringImplicit(matchValue(2)))
                        .bind("ln2_pattern")))
      .bind("ln2");
}

auto machterLn10() {
  return expr(anyOf(matchFloatLiteralNear("ln10_literal", llvm::numbers::ln10),
                    matchMathCall("log", ignoringImplicit(matchValue(10)))
                        .bind("ln10_pattern")))
      .bind("ln10");
}

auto matchSqrt2() {
  return expr(
             anyOf(matchFloatLiteralNear("sqrt2_literal", llvm::numbers::sqrt2),
                   matchSqrt(matchValue(2)).bind("sqrt2_pattern")))
      .bind("sqrt2");
}

auto matchSqrt3() {
  return expr(
             anyOf(matchFloatLiteralNear("sqrt3_literal", llvm::numbers::sqrt3),
                   matchSqrt(matchValue(3)).bind("sqrt3_pattern")))
      .bind("sqrt3");
}

auto matchInvSqrt3() {
  return expr(
             anyOf(
                 matchFloatLiteralNear("inv_sqrt3_literal",
                                       llvm::numbers::inv_sqrt3),
                 match1Div(matchSqrt(matchValue(3))).bind("inv_sqrt3_pattern")))
      .bind("inv_sqrt3");
}

auto matchPhi() {
  const auto PhiFormula = binaryOperator(
      hasOperatorName("/"),
      hasLHS(parenExpr(has(binaryOperator(
          hasOperatorName("+"), hasEitherOperand(matchValue(1)),
          hasEitherOperand(matchMathCall("sqrt", matchValue(5))))))),
      hasRHS(matchValue(2)));
  return expr(anyOf(PhiFormula.bind("phi_pattern"),
                    matchFloatLiteralNear("phi_literal", llvm::numbers::phi)))
      .bind("phi");
}

std::string getCode(const StringRef Constant, const bool IsFloat,
                    const bool IsLongDouble) {
  if (IsFloat) {
    return ("std::numbers::" + Constant + "_v<float>").str();
  }
  if (IsLongDouble) {
    return ("std::numbers::" + Constant + "_v<long double>").str();
  }
  return ("std::numbers::" + Constant).str();
}

bool isRangeOfCompleteMacro(const clang::SourceRange &Range,
                            const clang::SourceManager &SM,
                            const clang::LangOptions &LO) {
  if (!Range.getBegin().isMacroID()) {
    return false;
  }
  if (!clang::Lexer::isAtStartOfMacroExpansion(Range.getBegin(), SM, LO)) {
    return false;
  }

  if (!Range.getEnd().isMacroID()) {
    return false;
  }

  if (!clang::Lexer::isAtEndOfMacroExpansion(Range.getEnd(), SM, LO)) {
    return false;
  }

  return true;
}

} // namespace

namespace clang::tidy::modernize {
UseStdNumbersCheck::UseStdNumbersCheck(const StringRef Name,
                                       ClangTidyContext *const Context)
    : ClangTidyCheck(Name, Context),
      IncludeInserter(Options.getLocalOrGlobal("IncludeStyle",
                                               utils::IncludeSorter::IS_LLVM),
                      areDiagsSelfContained()) {}

void UseStdNumbersCheck::registerMatchers(MatchFinder *Finder) {
  static const auto ConstantMatchers = {
      matchLog2Euler(), matchLog10Euler(), matchEulerTopLevel(), matchEgamma(),
      matchInvSqrtPi(), matchInvPi(),      matchPiTopLevel(),    matchLn2(),
      machterLn10(),    matchSqrt2(),      matchInvSqrt3(),      matchSqrt3(),
      matchPhi(),
  };

  Finder->addMatcher(
      expr(anyOfExhaustive(ConstantMatchers),
           unless(isInTemplateInstantiation()),
           unless(hasParent(expr(
               anyOf(implicitCastExpr(hasImplicitDestinationType(isFloating())),
                     explicitCastExpr(hasDestinationType(isFloating())))))),
           hasType(qualType(hasCanonicalType(hasCanonicalTypeUnqualified(anyOf(
               qualType(asString("float")).bind("float"),
               qualType(asString("double")),
               qualType(asString("long double")).bind("long double"))))))),
      this);
}

void UseStdNumbersCheck::check(const MatchFinder::MatchResult &Result) {
  /*
    List of all math constants in the `<numbers>` header
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

  // The ordering determines what constants are looked at first.
  // E.g. look at 'inv_sqrt3' before 'sqrt3' to be able to replace the larger
  // expression
  constexpr auto Constants = std::array<std::pair<StringRef, double>, 13>{
      std::pair{StringRef{"log2e"}, llvm::numbers::log2e},
      std::pair{StringRef{"log10e"}, llvm::numbers::log10e},
      std::pair{StringRef{"e"}, llvm::numbers::e},
      std::pair{StringRef{"egamma"}, llvm::numbers::egamma},
      std::pair{StringRef{"inv_sqrtpi"}, llvm::numbers::inv_sqrtpi},
      std::pair{StringRef{"inv_pi"}, llvm::numbers::inv_pi},
      std::pair{StringRef{"pi"}, llvm::numbers::pi},
      std::pair{StringRef{"ln2"}, llvm::numbers::ln2},
      std::pair{StringRef{"ln10"}, llvm::numbers::ln10},
      std::pair{StringRef{"sqrt2"}, llvm::numbers::sqrt2},
      std::pair{StringRef{"inv_sqrt3"}, llvm::numbers::inv_sqrt3},
      std::pair{StringRef{"sqrt3"}, llvm::numbers::sqrt3},
      std::pair{StringRef{"phi"}, llvm::numbers::phi},
  };

  auto MatchedLiterals =
      llvm::SmallVector<std::tuple<std::string, double, const Expr *>>{};

  const auto &SM = *Result.SourceManager;
  const auto &LO = Result.Context->getLangOpts();

  const auto IsFloat = Result.Nodes.getNodeAs<QualType>("float") != nullptr;
  const auto IsLongDouble =
      Result.Nodes.getNodeAs<QualType>("long double") != nullptr;

  for (const auto &[ConstantName, ConstantValue] : Constants) {
    const auto *const Match = Result.Nodes.getNodeAs<Expr>(ConstantName);
    if (Match == nullptr) {
      continue;
    }

    const auto Range = Match->getSourceRange();

    if (Range.getBegin().isMacroID() &&
        !isRangeOfCompleteMacro(Range, SM, LO)) {
      continue;
    }

    const auto PatternBindString = (ConstantName + "_pattern").str();
    if (Result.Nodes.getNodeAs<Expr>(PatternBindString) != nullptr) {
      const auto Code = getCode(ConstantName, IsFloat, IsLongDouble);
      diag(Range.getBegin(), "prefer '%0' math constant")
          << Code << FixItHint::CreateReplacement(Range, Code);
      return;
    }

    const auto LiteralBindString = (ConstantName + "_literal").str();
    if (const auto *const Literal =
            Result.Nodes.getNodeAs<FloatingLiteral>(LiteralBindString)) {
      MatchedLiterals.emplace_back(
          ConstantName,
          std::abs(Literal->getValueAsApproximateDouble() - ConstantValue),
          Match);
    }
  }

  // We may have had no matches with literals, but a match with a pattern that
  // was a subexpression of a macro which was therefore skipped.
  if (MatchedLiterals.empty()) {
    return;
  }

  llvm::sort(MatchedLiterals, [](const auto &LHS, const auto &RHS) {
    return std::get<1>(LHS) < std::get<1>(RHS);
  });

  const auto &[Constant, _, Node] = MatchedLiterals.front();

  const auto Range = Node->getSourceRange();
  if (Range.getBegin().isMacroID() && !isRangeOfCompleteMacro(Range, SM, LO)) {
    return;
  }

  const auto Code = getCode(Constant, IsFloat, IsLongDouble);
  diag(Range.getBegin(), "prefer '%0' math constant")
      << Code << FixItHint::CreateReplacement(Range, Code)
      << IncludeInserter.createIncludeInsertion(
             Result.SourceManager->getFileID(Range.getBegin()), "<numbers>");
}

void UseStdNumbersCheck::registerPPCallbacks(const SourceManager &SM,
                                             Preprocessor *PP,
                                             Preprocessor *ModuleExpanderPP) {
  IncludeInserter.registerPreprocessor(PP);
}
} // namespace clang::tidy::modernize
