//===--- UseStdNumbersCheck.cpp - clang_tidy ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX_License_Identifier: Apache_2.0 WITH LLVM_exception
//
//===----------------------------------------------------------------------===//

#include "UseStdNumbersCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchersInternal.h"
#include "clang/ASTMatchers/ASTMatchersMacros.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/Token.h"
#include "clang/Tooling/Transformer/RewriteRule.h"
#include "clang/Tooling/Transformer/Stencil.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MathExtras.h"
#include <cstdint>
#include <string>

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

constexpr double Pi = 3.141592653589793238462643383279502884197169399;
constexpr double Euler = 2.718281828459045235360287471352662497757247093;
constexpr double Phi = 1.618033988749894848204586834365638117720309179;
constexpr double Egamma = 0.577215664901532860606512090082402431042159335;

constexpr auto DiffThreshold = 0.001;

AST_MATCHER_P2(clang::FloatingLiteral, near, double, Value, double, Threshold) {
  return std::abs(Node.getValue().convertToDouble() - Value) < Threshold;
}

// We don't want to match uses of macros, such as
//
// auto PiHalved = MY_PI / 2;
//
// because a project might use defines for math constants.
// Instead, the macro definition is matched and the value is exchanged there.
// Hinting at replacing macro definitions with language constructs is done in
// another check.
AST_MATCHER(clang::Expr, isMathMacro) { return Node.getBeginLoc().isMacroID(); }

AST_MATCHER_P(clang::QualType, hasUnqualifiedDesugaredType,
              Matcher<clang::QualType>, InnerMatcher) {
  return InnerMatcher.matches(Node->getCanonicalTypeUnqualified(), Finder,
                              Builder);
}

auto matchMathCall(const StringRef FunctionName,
                   const Matcher<clang::Expr> ArgumentMatcher) {
  return callExpr(callee(functionDecl(hasName(FunctionName))),
                  hasArgument(0, ignoringImplicit(ArgumentMatcher)));
}

auto matchSqrt(const Matcher<clang::Expr> ArgumentMatcher) {
  return matchMathCall("sqrt", ArgumentMatcher);
}

// 'MatchDeclRefExprOrMacro' is used to differentiate matching expressions where
// the value of anything used is near 'Val' and matching expressions where we
// only care about the actual literal.
// We don't want top-level matches to match a simple DeclRefExpr/macro that was
// initialized with this value because projects might declare their own
// constants (e.g. namespaced constants or macros) to be used. We don't want to
// flag the use of these variables/constants, but modify the definition of the
// variable or macro.
//
// example:
//   const auto e = 2.71828182; // std::numbers::e
//                  ^^^^^^^^^^
//                  match here
//
//   auto use = e / 2;
//              ^
//   don't match this as a top-level match, this would create noise
//
//   auto use2 = log2(e); // std::numbers::log2e
//               ^^^^^^^
//               match here, matcher needs to check the initialization
//               of e to match log2e
//
// Therefore, all top-level matcher set MatchDeclRefExprOrMacro to false
auto matchFloatValueNear(const double Val,
                         const bool MatchDeclRefExprOrMacro = true) {
  const auto FloatVal = floatLiteral(near(Val, DiffThreshold));
  if (!MatchDeclRefExprOrMacro) {
    return expr(unless(isMathMacro()), ignoringImplicit(FloatVal));
  }

  const auto Dref = declRefExpr(to(varDecl(
      anyOf(isConstexpr(), varDecl(hasType(qualType(isConstQualified())))),
      hasInitializer(FloatVal))));
  return expr(ignoringImplicit(anyOf(FloatVal, Dref)));
}

auto matchValue(const int64_t ValInt) {
  const auto Int2 = integerLiteral(equals(ValInt));
  const auto Float2 = matchFloatValueNear(static_cast<double>(ValInt));
  const auto Dref = declRefExpr(to(varDecl(
      anyOf(isConstexpr(), varDecl(hasType(qualType(isConstQualified())))),
      hasInitializer(expr(ignoringImplicit(anyOf(Int2, Float2)))))));
  return expr(ignoringImplicit(anyOf(Int2, Float2, Dref)));
}

auto match1Div(const Matcher<clang::Expr> Match) {
  return binaryOperator(hasOperatorName("/"), hasLHS(matchValue(1)),
                        hasRHS(ignoringImplicit(Match)));
}

auto matchEuler() {
  return expr(
      anyOf(matchFloatValueNear(Euler), matchMathCall("exp", matchValue(1))));
}
auto matchEulerTopLevel() {
  return expr(anyOf(matchFloatValueNear(Euler, false),
                    matchMathCall("exp", matchValue(1))));
}

auto matchLog2Euler() {
  return expr(anyOf(matchFloatValueNear(llvm::numbers::log2e, false),
                    matchMathCall("log2", matchEuler())));
}

auto matchLog10Euler() {
  return expr(anyOf(matchFloatValueNear(llvm::numbers::log10e, false),
                    matchMathCall("log10", matchEuler())));
}

auto matchPi() { return matchFloatValueNear(Pi); }
auto matchPiTopLevel() { return matchFloatValueNear(Pi, false); }

auto matchEgamma() { return matchFloatValueNear(Egamma, false); }

auto matchInvPi() {
  return expr(
      anyOf(matchFloatValueNear(llvm::numbers::inv_pi), match1Div(matchPi())));
}

auto matchInvSqrtPi() {
  return expr(anyOf(matchFloatValueNear(llvm::numbers::inv_sqrtpi, false),
                    match1Div(matchSqrt(matchPi()))));
}

auto matchLn2() {
  return expr(anyOf(matchFloatValueNear(llvm::numbers::ln2, false),
                    matchMathCall("log", ignoringImplicit(matchValue(2)))));
}

auto machterLn10() {
  return expr(anyOf(matchFloatValueNear(llvm::numbers::ln10, false),
                    matchMathCall("log", ignoringImplicit(matchValue(10)))));
}

auto matchSqrt2() {
  return expr(anyOf(matchFloatValueNear(llvm::numbers::sqrt2, false),
                    matchSqrt(matchValue(2))));
}

auto matchSqrt3() {
  return expr(anyOf(matchFloatValueNear(llvm::numbers::sqrt3, false),
                    matchSqrt(matchValue(3))));
}

auto matchInvSqrt3() {
  return expr(anyOf(matchFloatValueNear(llvm::numbers::inv_sqrt3, false),
                    match1Div(matchSqrt(matchValue(3)))));
}

auto matchPhi() {
  const auto PhiFormula = binaryOperator(
      hasOperatorName("/"),
      hasLHS(parenExpr(has(binaryOperator(
          hasOperatorName("+"), hasEitherOperand(matchValue(1)),
          hasEitherOperand(matchMathCall("sqrt", matchValue(5))))))),
      hasRHS(matchValue(2)));
  return expr(anyOf(PhiFormula, matchFloatValueNear(Phi)));
}

EditGenerator
chainedIfBound(ASTEdit DefaultEdit,
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

  const auto EditRules = chainedIfBound(
      DefaultEdit, {{"float", FloatEdit}, {"long double", LongDoubleEdit}});

  return makeRule(
      expr(Matcher,
           hasType(qualType(hasCanonicalType(hasUnqualifiedDesugaredType(anyOf(
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

class MathConstantMacroCallback : public clang::PPCallbacks {
public:
  explicit MathConstantMacroCallback(
      clang::tidy::modernize::UseStdNumbersCheck *Check)
      : Check{Check} {};

  void MacroDefined(const clang::Token & /*MacroNameTok*/,
                    const clang::MacroDirective *MD) override {
    for (const auto &Tok : MD->getDefinition().getMacroInfo()->tokens()) {
      if (!Tok.is(clang::tok::numeric_constant)) {
        continue;
      }

      const auto Definition =
          llvm::StringRef{Tok.getLiteralData(), Tok.getLength()};
      double Value{};
      Definition.getAsDouble(Value);

      const auto IsNear = [](const auto Lhs, const auto Rhs) {
        return std::abs(Lhs - Rhs) < DiffThreshold;
      };

      if (IsNear(Value, llvm::numbers::log2e)) {
        reportDiag(Tok, "std::numbers::log2e");
        return;
      }
      if (IsNear(Value, llvm::numbers::log10e)) {
        reportDiag(Tok, "std::numbers::log10e");
        return;
      }
      if (IsNear(Value, llvm::numbers::e)) {
        reportDiag(Tok, "std::numbers::e");
        return;
      }
      if (IsNear(Value, llvm::numbers::egamma)) {
        reportDiag(Tok, "std::numbers::egamma");
        return;
      }
      if (IsNear(Value, llvm::numbers::inv_sqrtpi)) {
        reportDiag(Tok, "std::numbers::inv_sqrtpi");
        return;
      }
      if (IsNear(Value, llvm::numbers::inv_pi)) {
        reportDiag(Tok, "std::numbers::inv_pi");
        return;
      }
      if (IsNear(Value, llvm::numbers::pi)) {
        reportDiag(Tok, "std::numbers::pi");
        return;
      }
      if (IsNear(Value, llvm::numbers::ln2)) {
        reportDiag(Tok, "std::numbers::ln2");
        return;
      }
      if (IsNear(Value, llvm::numbers::ln10)) {
        reportDiag(Tok, "std::numbers::ln10");
        return;
      }
      if (IsNear(Value, llvm::numbers::sqrt2)) {
        reportDiag(Tok, "std::numbers::sqrt2");
        return;
      }
      if (IsNear(Value, llvm::numbers::inv_sqrt3)) {
        reportDiag(Tok, "std::numbers::inv_sqrt3");
        return;
      }
      if (IsNear(Value, llvm::numbers::sqrt3)) {
        reportDiag(Tok, "std::numbers::sqrt3");
        return;
      }
      if (IsNear(Value, llvm::numbers::phi)) {
        reportDiag(Tok, "std::numbers::phi");
        return;
      }
    }
  }

private:
  void reportDiag(const clang::Token &Tok, const llvm::StringRef Constant) {
    Check->diag(Tok.getLocation(), "prefer math constant")
        << clang::FixItHint::CreateReplacement(
               clang::SourceRange(Tok.getLocation(), Tok.getLastLoc()),
               Constant);
  }

  clang::tidy::modernize::UseStdNumbersCheck *Check{};
};
} // namespace

namespace clang::tidy::modernize {
UseStdNumbersCheck::UseStdNumbersCheck(const StringRef Name,
                                       ClangTidyContext *const Context)
    : TransformerClangTidyCheck(Name, Context) {
  setRule(makeRewriteRule());
}

void UseStdNumbersCheck::registerPPCallbacks(const SourceManager &SM,
                                             Preprocessor *PP,
                                             Preprocessor *ModuleExpanderPP) {
  utils::TransformerClangTidyCheck::registerPPCallbacks(SM, PP,
                                                        ModuleExpanderPP);
  PP->addPPCallbacks(std::make_unique<MathConstantMacroCallback>(this));
}
} // namespace clang::tidy::modernize
