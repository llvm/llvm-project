//===--- UseBuiltinLiteralsCheck.cpp - clang-tidy -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseBuiltinLiteralsCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <optional>

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

namespace {

using RuleOnStd = bool (*)(const LangStandard &LS);

struct Replacement {
  Replacement(StringRef Seq, const RuleOnStd Std = nullptr)
      : Seq(Seq), Std(Std) {}
  bool operator()(const LangOptions &LO) const {
    return Std ? Std(LangStandard::getLangStandardForKind(LO.LangStd)) : true;
  }
  StringRef Seq;
  RuleOnStd Std;
};

} // namespace

static const llvm::Regex CharRegex("^(u8|u|U|L)?");
static const llvm::StringMap<Replacement> CharPrefix({
    {"char", {""}},
    {"char8_t", {"u8"}},
    {"char16_t", {"u"}},
    {"char32_t", {"U"}},
    {"wchar_t", {"L"}},
});

static const llvm::Regex
    IntRegex("(([uU]?[lL]{0,2})|([lL]{0,2}[uU]?)|([uU]?[zZ]?)|([zZ]?[uU]?))?$");
static const llvm::StringMap<Replacement> IntSuffix({
    {"int", {""}},
    {"unsigned int", {"u"}},
    {"long", {"L"}},
    {"unsigned long", {"uL"}},
    {"long long", {"LL"}},
    {"unsigned long long", {"uLL"}},
    {"size_t", {"uz", [](const auto &LS) { return LS.isCPlusPlus23(); }}},
    {"std::size_t", {"uz", [](const auto &LS) { return LS.isCPlusPlus23(); }}},
});

static const llvm::Regex FloatRegex(
    "([fF]|[lL]|([fF]16)|([fF]32)|([fF]64)|([fF]128)|((bf|BF)16))?$");
static const llvm::StringMap<Replacement> FloatSuffix({
    {"double", {""}},
    {"float", {"f"}},
    {"long double", {"L"}},
    {"std::float16_t", {"f16"}},
    {"std::float32_t", {"f32"}},
    {"std::float64_t", {"f64"}},
    {"std::float128_t", {"f128"}},
    {"std::bfloat16_t", {"bf16"}},
    {"float16_t", {"f16"}},
    {"float32_t", {"f32"}},
    {"float64_t", {"f64"}},
    {"float128_t", {"f128"}},
    {"bfloat16_t", {"bf16"}},
});

void UseBuiltinLiteralsCheck::registerMatchers(MatchFinder *Finder) {
  static const auto Literal = has(ignoringParenImpCasts(
      expr(anyOf(characterLiteral().bind("char"), integerLiteral().bind("int"),
                 floatLiteral().bind("float")))
          .bind("lit")));
  Finder->addMatcher(
      traverse(TK_IgnoreUnlessSpelledInSource,
               explicitCastExpr(anyOf(Literal, has(initListExpr(Literal))))
                   .bind("expr")),
      this);
}

static StringRef getRawStringRef(const SourceRange &Range,
                                 const SourceManager &Sources,
                                 const LangOptions &LangOpts) {
  CharSourceRange TextRange = Lexer::getAsCharRange(Range, Sources, LangOpts);
  return Lexer::getSourceText(TextRange, Sources, LangOpts);
}

void UseBuiltinLiteralsCheck::check(const MatchFinder::MatchResult &Result) {

  const auto &SM = *Result.SourceManager;
  const auto &Nodes = Result.Nodes;

  const auto *MatchedCast = Nodes.getNodeAs<ExplicitCastExpr>("expr");
  const auto *Lit = Nodes.getNodeAs<Expr>("lit");
  assert(MatchedCast && Lit);

  StringRef LitText = getRawStringRef(Lit->getExprLoc(), SM, getLangOpts());
  std::string CastType = MatchedCast->getTypeAsWritten().getAsString();
  std::string Fix; // Replacement string for the fix-it hint.
  std::optional<StringRef> Seq; // Literal sequence, prefix or suffix.

  if (const auto *CharLit = Nodes.getNodeAs<CharacterLiteral>("char");
      CharLit && CharPrefix.contains(CastType)) {
    if (const Replacement &Rep = CharPrefix.at(CastType); Rep(getLangOpts())) {

      Seq = Rep.Seq;
      if (!CharLit->getLocation().isMacroID()) {
        Fix.append(Rep.Seq);
        Fix.append(CharRegex.sub("", LitText.str()));
      }
    }
  } else if (const auto *IntLit = Nodes.getNodeAs<IntegerLiteral>("int");
             IntLit && IntSuffix.contains(CastType)) {
    if (const Replacement &Rep = IntSuffix.at(CastType); Rep(getLangOpts())) {

      Seq = Rep.Seq;
      if (!IntLit->getLocation().isMacroID()) {
        Fix.append(IntRegex.sub("", LitText.str()));
        Fix.append(Rep.Seq);
      }
    }
  } else if (const auto *FloatLit = Nodes.getNodeAs<FloatingLiteral>("float");
             FloatLit && FloatSuffix.contains(CastType)) {
    if (const Replacement &Rep = FloatSuffix.at(CastType); Rep(getLangOpts())) {

      Seq = Rep.Seq;
      if (!FloatLit->getLocation().isMacroID()) {
        Fix.append(FloatRegex.sub("", LitText.str()));
        Fix.append(Rep.Seq);
      }
    }
  }

  const TypeLoc CastTypeLoc = MatchedCast->getTypeInfoAsWritten()->getTypeLoc();

  if (!Fix.empty() && !CastTypeLoc.getBeginLoc().isMacroID()) {

    // Recommend fix-it when no part of the explicit cast comes from a macro.
    diag(MatchedCast->getExprLoc(),
         "use built-in literal instead of explicit cast")
        << FixItHint::CreateReplacement(MatchedCast->getSourceRange(), Fix);
  } else if (Seq && MatchedCast->getExprLoc().isMacroID()) {

    // Recommend manual fix when the entire explicit cast is within a macro.
    diag(MatchedCast->getExprLoc(),
         "use built-in '%0' %1 instead of explicit cast to '%2'")
        << *Seq
		<< (Nodes.getNodeAs<CharacterLiteral>("char") ? "prefix" : "suffix")
		<< CastType;
  }
}

} // namespace clang::tidy::readability
