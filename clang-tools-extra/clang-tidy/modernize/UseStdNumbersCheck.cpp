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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
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

AST_MATCHER_P2(clang::FloatingLiteral, near, double, Value, double,
               DiffThreshold) {
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

AST_MATCHER_P(clang::Expr, anyOfExhaustive, std::vector<Matcher<clang::Stmt>>,
              Exprs) {
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

// Using this struct to store the 'DiffThreshold' config value to create the
// matchers without the need to pass 'DiffThreshold' into every matcher.
// 'DiffThreshold' is needed in the 'near' matcher, which is used for matching
// the literal of every constant and for formulas' subexpressions that look at
// literals.
struct MatchBuilder {
  auto
  ignoreParenAndArithmeticCasting(const Matcher<clang::Expr> Matcher) const {
    return expr(hasType(qualType(isArithmetic())), ignoringParenCasts(Matcher));
  }

  auto ignoreParenAndFloatingCasting(const Matcher<clang::Expr> Matcher) const {
    return expr(hasType(qualType(isFloating())), ignoringParenCasts(Matcher));
  }

  auto matchMathCall(const StringRef FunctionName,
                     const Matcher<clang::Expr> ArgumentMatcher) const {
    return expr(ignoreParenAndFloatingCasting(
        callExpr(callee(functionDecl(hasName(FunctionName),
                                     hasParameter(0, hasType(isArithmetic())))),
                 hasArgument(0, ArgumentMatcher))));
  }

  auto matchSqrt(const Matcher<clang::Expr> ArgumentMatcher) const {
    return matchMathCall("sqrt", ArgumentMatcher);
  }

  // Used for top-level matchers (i.e. the match that replaces Val with its
  // constant).
  //
  // E.g. The matcher of `std::numbers::pi` uses this matcher to look for
  // floatLiterals that have the value of pi.
  //
  // If the match is for a top-level match, we only care about the literal.
  auto matchFloatLiteralNear(const StringRef Constant, const double Val) const {
    return expr(ignoreParenAndFloatingCasting(
        floatLiteral(near(Val, DiffThreshold)).bind(Constant)));
  }

  // Used for non-top-level matchers (i.e. matchers that are used as inner
  // matchers for top-level matchers).
  //
  // E.g.: The matcher of `std::numbers::log2e` uses this matcher to check if
  // `e` of `log2(e)` is declared constant and initialized with the value for
  // eulers number.
  //
  // Here, we do care about literals and about DeclRefExprs to variable
  // declarations that are constant and initialized with `Val`. This allows
  // top-level matchers to see through declared constants for their inner
  // matches like the `std::numbers::log2e` matcher.
  auto matchFloatValueNear(const double Val) const {
    const auto Float = floatLiteral(near(Val, DiffThreshold));

    const auto Dref = declRefExpr(
        to(varDecl(hasType(qualType(isConstQualified(), isFloating())),
                   hasInitializer(ignoreParenAndFloatingCasting(Float)))));
    return expr(ignoreParenAndFloatingCasting(anyOf(Float, Dref)));
  }

  auto matchValue(const int64_t ValInt) const {
    const auto Int =
        expr(ignoreParenAndArithmeticCasting(integerLiteral(equals(ValInt))));
    const auto Float = expr(ignoreParenAndFloatingCasting(
        matchFloatValueNear(static_cast<double>(ValInt))));
    const auto Dref = declRefExpr(to(varDecl(
        hasType(qualType(isConstQualified(), isArithmetic())),
        hasInitializer(expr(anyOf(ignoringImplicit(Int),
                                  ignoreParenAndFloatingCasting(Float)))))));
    return expr(anyOf(Int, Float, Dref));
  }

  auto match1Div(const Matcher<clang::Expr> Match) const {
    return binaryOperator(hasOperatorName("/"), hasLHS(matchValue(1)),
                          hasRHS(Match));
  }

  auto matchEuler() const {
    return expr(anyOf(matchFloatValueNear(llvm::numbers::e),
                      matchMathCall("exp", matchValue(1))));
  }
  auto matchEulerTopLevel() const {
    return expr(anyOf(matchFloatLiteralNear("e_literal", llvm::numbers::e),
                      matchMathCall("exp", matchValue(1)).bind("e_pattern")))
        .bind("e");
  }

  auto matchLog2Euler() const {
    return expr(
               anyOf(
                   matchFloatLiteralNear("log2e_literal", llvm::numbers::log2e),
                   matchMathCall("log2", matchEuler()).bind("log2e_pattern")))
        .bind("log2e");
  }

  auto matchLog10Euler() const {
    return expr(
               anyOf(
                   matchFloatLiteralNear("log10e_literal",
                                         llvm::numbers::log10e),
                   matchMathCall("log10", matchEuler()).bind("log10e_pattern")))
        .bind("log10e");
  }

  auto matchPi() const { return matchFloatValueNear(llvm::numbers::pi); }
  auto matchPiTopLevel() const {
    return matchFloatLiteralNear("pi_literal", llvm::numbers::pi).bind("pi");
  }

  auto matchEgamma() const {
    return matchFloatLiteralNear("egamma_literal", llvm::numbers::egamma)
        .bind("egamma");
  }

  auto matchInvPi() const {
    return expr(anyOf(matchFloatLiteralNear("inv_pi_literal",
                                            llvm::numbers::inv_pi),
                      match1Div(matchPi()).bind("inv_pi_pattern")))
        .bind("inv_pi");
  }

  auto matchInvSqrtPi() const {
    return expr(anyOf(
                    matchFloatLiteralNear("inv_sqrtpi_literal",
                                          llvm::numbers::inv_sqrtpi),
                    match1Div(matchSqrt(matchPi())).bind("inv_sqrtpi_pattern")))
        .bind("inv_sqrtpi");
  }

  auto matchLn2() const {
    return expr(anyOf(matchFloatLiteralNear("ln2_literal", llvm::numbers::ln2),
                      matchMathCall("log", matchValue(2)).bind("ln2_pattern")))
        .bind("ln2");
  }

  auto machterLn10() const {
    return expr(
               anyOf(matchFloatLiteralNear("ln10_literal", llvm::numbers::ln10),
                     matchMathCall("log", matchValue(10)).bind("ln10_pattern")))
        .bind("ln10");
  }

  auto matchSqrt2() const {
    return expr(anyOf(matchFloatLiteralNear("sqrt2_literal",
                                            llvm::numbers::sqrt2),
                      matchSqrt(matchValue(2)).bind("sqrt2_pattern")))
        .bind("sqrt2");
  }

  auto matchSqrt3() const {
    return expr(anyOf(matchFloatLiteralNear("sqrt3_literal",
                                            llvm::numbers::sqrt3),
                      matchSqrt(matchValue(3)).bind("sqrt3_pattern")))
        .bind("sqrt3");
  }

  auto matchInvSqrt3() const {
    return expr(anyOf(matchFloatLiteralNear("inv_sqrt3_literal",
                                            llvm::numbers::inv_sqrt3),
                      match1Div(matchSqrt(matchValue(3)))
                          .bind("inv_sqrt3_pattern")))
        .bind("inv_sqrt3");
  }

  auto matchPhi() const {
    const auto PhiFormula = binaryOperator(
        hasOperatorName("/"),
        hasLHS(binaryOperator(
            hasOperatorName("+"), hasEitherOperand(matchValue(1)),
            hasEitherOperand(matchMathCall("sqrt", matchValue(5))))),
        hasRHS(matchValue(2)));
    return expr(anyOf(PhiFormula.bind("phi_pattern"),
                      matchFloatLiteralNear("phi_literal", llvm::numbers::phi)))
        .bind("phi");
  }

  double DiffThreshold;
};

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
                      areDiagsSelfContained()),
      DiffThresholdString{Options.get("DiffThreshold", "0.001")} {
  if (DiffThresholdString.getAsDouble(DiffThreshold)) {
    configurationDiag(
        "Invalid DiffThreshold config value: '%0', expected a double")
        << DiffThresholdString;
    DiffThreshold = 0.001;
  }
}

void UseStdNumbersCheck::registerMatchers(MatchFinder *const Finder) {
  const auto Matches = MatchBuilder{DiffThreshold};
  std::vector<Matcher<clang::Stmt>> ConstantMatchers = {
      Matches.matchLog2Euler(),     Matches.matchLog10Euler(),
      Matches.matchEulerTopLevel(), Matches.matchEgamma(),
      Matches.matchInvSqrtPi(),     Matches.matchInvPi(),
      Matches.matchPiTopLevel(),    Matches.matchLn2(),
      Matches.machterLn10(),        Matches.matchSqrt2(),
      Matches.matchInvSqrt3(),      Matches.matchSqrt3(),
      Matches.matchPhi(),
  };

  Finder->addMatcher(
      expr(
          anyOfExhaustive(std::move(ConstantMatchers)),
          unless(hasParent(explicitCastExpr(hasDestinationType(isFloating())))),
          hasType(qualType(hasCanonicalTypeUnqualified(
              anyOf(qualType(asString("float")).bind("float"),
                    qualType(asString("double")),
                    qualType(asString("long double")).bind("long double")))))),
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

    const auto IsMacro = Range.getBegin().isMacroID();

    // We do not want to emit a diagnostic when we are matching a macro, but the
    // match inside of the macro does not cover the whole macro.
    if (IsMacro && !isRangeOfCompleteMacro(Range, SM, LO)) {
      continue;
    }

    if (const auto PatternBindString = (ConstantName + "_pattern").str();
        Result.Nodes.getNodeAs<Expr>(PatternBindString) != nullptr) {
      const auto Code = getCode(ConstantName, IsFloat, IsLongDouble);
      diag(Range.getBegin(), "prefer '%0' to this %select{formula|macro}1")
          << Code << IsMacro << FixItHint::CreateReplacement(Range, Code);
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
  // was a part of a macro which was therefore skipped.
  if (MatchedLiterals.empty()) {
    return;
  }

  llvm::sort(MatchedLiterals, [](const auto &LHS, const auto &RHS) {
    return std::get<1>(LHS) < std::get<1>(RHS);
  });

  const auto &[Constant, Diff, Node] = MatchedLiterals.front();

  const auto Range = Node->getSourceRange();
  const auto IsMacro = Range.getBegin().isMacroID();

  // We do not want to emit a diagnostic when we are matching a macro, but the
  // match inside of the macro does not cover the whole macro.
  if (IsMacro && !isRangeOfCompleteMacro(Range, SM, LO)) {
    return;
  }

  const auto Code = getCode(Constant, IsFloat, IsLongDouble);
  diag(Range.getBegin(),
       "prefer '%0' to this %select{literal|macro}1, differs by '%2'")
      << Code << IsMacro << llvm::formatv("{0:e2}", Diff).str()
      << FixItHint::CreateReplacement(Range, Code)
      << IncludeInserter.createIncludeInsertion(
             Result.SourceManager->getFileID(Range.getBegin()), "<numbers>");
}

void UseStdNumbersCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *const PP,
    Preprocessor *const ModuleExpanderPP) {
  IncludeInserter.registerPreprocessor(PP);
}

void UseStdNumbersCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IncludeStyle", IncludeInserter.getStyle());
  Options.store(Opts, "DiffThreshold", DiffThresholdString);
}
} // namespace clang::tidy::modernize
