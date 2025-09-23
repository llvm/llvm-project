//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OperatorsRepresentationCheck.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/STLExtras.h"
#include <array>
#include <utility>

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

static StringRef getOperatorSpelling(SourceLocation Loc, ASTContext &Context) {
  if (Loc.isInvalid())
    return {};

  SourceManager &SM = Context.getSourceManager();

  Loc = SM.getSpellingLoc(Loc);
  if (Loc.isInvalid())
    return {};

  const CharSourceRange TokenRange = CharSourceRange::getTokenRange(Loc);
  return Lexer::getSourceText(TokenRange, SM, Context.getLangOpts());
}

namespace {

AST_MATCHER_P2(BinaryOperator, hasInvalidBinaryOperatorRepresentation,
               BinaryOperatorKind, Kind, llvm::StringRef,
               ExpectedRepresentation) {
  if (Node.getOpcode() != Kind || ExpectedRepresentation.empty())
    return false;

  StringRef Spelling =
      getOperatorSpelling(Node.getOperatorLoc(), Finder->getASTContext());
  return !Spelling.empty() && Spelling != ExpectedRepresentation;
}

AST_MATCHER_P2(UnaryOperator, hasInvalidUnaryOperatorRepresentation,
               UnaryOperatorKind, Kind, llvm::StringRef,
               ExpectedRepresentation) {
  if (Node.getOpcode() != Kind || ExpectedRepresentation.empty())
    return false;

  StringRef Spelling =
      getOperatorSpelling(Node.getOperatorLoc(), Finder->getASTContext());
  return !Spelling.empty() && Spelling != ExpectedRepresentation;
}

AST_MATCHER_P2(CXXOperatorCallExpr, hasInvalidOverloadedOperatorRepresentation,
               OverloadedOperatorKind, Kind, llvm::StringRef,
               ExpectedRepresentation) {
  if (Node.getOperator() != Kind || ExpectedRepresentation.empty())
    return false;

  StringRef Spelling =
      getOperatorSpelling(Node.getOperatorLoc(), Finder->getASTContext());
  return !Spelling.empty() && Spelling != ExpectedRepresentation;
}

} // namespace

constexpr std::array<std::pair<llvm::StringRef, llvm::StringRef>, 2U>
    UnaryRepresentation{{{"!", "not"}, {"~", "compl"}}};

constexpr std::array<std::pair<llvm::StringRef, llvm::StringRef>, 9U>
    OperatorsRepresentation{{{"&&", "and"},
                             {"||", "or"},
                             {"^", "xor"},
                             {"&", "bitand"},
                             {"|", "bitor"},
                             {"&=", "and_eq"},
                             {"|=", "or_eq"},
                             {"!=", "not_eq"},
                             {"^=", "xor_eq"}}};

static llvm::StringRef translate(llvm::StringRef Value) {
  for (const auto &[Traditional, Alternative] : UnaryRepresentation) {
    if (Value == Traditional)
      return Alternative;
    if (Value == Alternative)
      return Traditional;
  }

  for (const auto &[Traditional, Alternative] : OperatorsRepresentation) {
    if (Value == Traditional)
      return Alternative;
    if (Value == Alternative)
      return Traditional;
  }
  return {};
}

static bool isNotOperatorStr(llvm::StringRef Value) {
  return translate(Value).empty();
}

static bool isSeparator(char C) noexcept {
  constexpr llvm::StringRef Separators(" \t\r\n\0()<>{};,");
  return Separators.contains(C);
}

static bool needEscaping(llvm::StringRef Operator) {
  switch (Operator[0]) {
  case '&':
  case '|':
  case '!':
  case '^':
  case '~':
    return false;
  default:
    return true;
  }
}

static llvm::StringRef
getRepresentation(const std::vector<llvm::StringRef> &Config,
                  llvm::StringRef Traditional, llvm::StringRef Alternative) {
  if (llvm::is_contained(Config, Traditional))
    return Traditional;
  if (llvm::is_contained(Config, Alternative))
    return Alternative;
  return {};
}

template <typename T>
static bool isAnyOperatorEnabled(const std::vector<llvm::StringRef> &Config,
                                 const T &Operators) {
  for (const auto &[traditional, alternative] : Operators) {
    if (!getRepresentation(Config, traditional, alternative).empty())
      return true;
  }
  return false;
}

OperatorsRepresentationCheck::OperatorsRepresentationCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      BinaryOperators(
          utils::options::parseStringList(Options.get("BinaryOperators", ""))),
      OverloadedOperators(utils::options::parseStringList(
          Options.get("OverloadedOperators", ""))) {
  llvm::erase_if(BinaryOperators, isNotOperatorStr);
  llvm::erase_if(OverloadedOperators, isNotOperatorStr);
}

void OperatorsRepresentationCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "BinaryOperators",
                utils::options::serializeStringList(BinaryOperators));
  Options.store(Opts, "OverloadedOperators",
                utils::options::serializeStringList(OverloadedOperators));
}

std::optional<TraversalKind>
OperatorsRepresentationCheck::getCheckTraversalKind() const {
  return TK_IgnoreUnlessSpelledInSource;
}

bool OperatorsRepresentationCheck::isLanguageVersionSupported(
    const LangOptions &LangOpts) const {
  return LangOpts.CPlusPlus;
}

void OperatorsRepresentationCheck::registerBinaryOperatorMatcher(
    MatchFinder *Finder) {
  if (!isAnyOperatorEnabled(BinaryOperators, OperatorsRepresentation))
    return;

  Finder->addMatcher(
      binaryOperator(
          unless(isExpansionInSystemHeader()),
          anyOf(hasInvalidBinaryOperatorRepresentation(
                    BO_LAnd, getRepresentation(BinaryOperators, "&&", "and")),
                hasInvalidBinaryOperatorRepresentation(
                    BO_LOr, getRepresentation(BinaryOperators, "||", "or")),
                hasInvalidBinaryOperatorRepresentation(
                    BO_NE, getRepresentation(BinaryOperators, "!=", "not_eq")),
                hasInvalidBinaryOperatorRepresentation(
                    BO_Xor, getRepresentation(BinaryOperators, "^", "xor")),
                hasInvalidBinaryOperatorRepresentation(
                    BO_And, getRepresentation(BinaryOperators, "&", "bitand")),
                hasInvalidBinaryOperatorRepresentation(
                    BO_Or, getRepresentation(BinaryOperators, "|", "bitor")),
                hasInvalidBinaryOperatorRepresentation(
                    BO_AndAssign,
                    getRepresentation(BinaryOperators, "&=", "and_eq")),
                hasInvalidBinaryOperatorRepresentation(
                    BO_OrAssign,
                    getRepresentation(BinaryOperators, "|=", "or_eq")),
                hasInvalidBinaryOperatorRepresentation(
                    BO_XorAssign,
                    getRepresentation(BinaryOperators, "^=", "xor_eq"))))
          .bind("binary_op"),
      this);
}

void OperatorsRepresentationCheck::registerUnaryOperatorMatcher(
    MatchFinder *Finder) {
  if (!isAnyOperatorEnabled(BinaryOperators, UnaryRepresentation))
    return;

  Finder->addMatcher(
      unaryOperator(
          unless(isExpansionInSystemHeader()),
          anyOf(hasInvalidUnaryOperatorRepresentation(
                    UO_LNot, getRepresentation(BinaryOperators, "!", "not")),
                hasInvalidUnaryOperatorRepresentation(
                    UO_Not, getRepresentation(BinaryOperators, "~", "compl"))))
          .bind("unary_op"),
      this);
}

void OperatorsRepresentationCheck::registerOverloadedOperatorMatcher(
    MatchFinder *Finder) {
  if (!isAnyOperatorEnabled(OverloadedOperators, OperatorsRepresentation) &&
      !isAnyOperatorEnabled(OverloadedOperators, UnaryRepresentation))
    return;

  Finder->addMatcher(
      cxxOperatorCallExpr(
          unless(isExpansionInSystemHeader()),
          anyOf(
              hasInvalidOverloadedOperatorRepresentation(
                  OO_AmpAmp,
                  getRepresentation(OverloadedOperators, "&&", "and")),
              hasInvalidOverloadedOperatorRepresentation(
                  OO_PipePipe,
                  getRepresentation(OverloadedOperators, "||", "or")),
              hasInvalidOverloadedOperatorRepresentation(
                  OO_Exclaim,
                  getRepresentation(OverloadedOperators, "!", "not")),
              hasInvalidOverloadedOperatorRepresentation(
                  OO_ExclaimEqual,
                  getRepresentation(OverloadedOperators, "!=", "not_eq")),
              hasInvalidOverloadedOperatorRepresentation(
                  OO_Caret, getRepresentation(OverloadedOperators, "^", "xor")),
              hasInvalidOverloadedOperatorRepresentation(
                  OO_Amp,
                  getRepresentation(OverloadedOperators, "&", "bitand")),
              hasInvalidOverloadedOperatorRepresentation(
                  OO_Pipe,
                  getRepresentation(OverloadedOperators, "|", "bitor")),
              hasInvalidOverloadedOperatorRepresentation(
                  OO_AmpEqual,
                  getRepresentation(OverloadedOperators, "&=", "and_eq")),
              hasInvalidOverloadedOperatorRepresentation(
                  OO_PipeEqual,
                  getRepresentation(OverloadedOperators, "|=", "or_eq")),
              hasInvalidOverloadedOperatorRepresentation(
                  OO_CaretEqual,
                  getRepresentation(OverloadedOperators, "^=", "xor_eq")),
              hasInvalidOverloadedOperatorRepresentation(
                  OO_Tilde,
                  getRepresentation(OverloadedOperators, "~", "compl"))))
          .bind("overloaded_op"),
      this);
}

void OperatorsRepresentationCheck::registerMatchers(MatchFinder *Finder) {
  registerBinaryOperatorMatcher(Finder);
  registerUnaryOperatorMatcher(Finder);
  registerOverloadedOperatorMatcher(Finder);
}

void OperatorsRepresentationCheck::check(
    const MatchFinder::MatchResult &Result) {

  SourceLocation Loc;

  if (const auto *Op = Result.Nodes.getNodeAs<BinaryOperator>("binary_op"))
    Loc = Op->getOperatorLoc();
  else if (const auto *Op = Result.Nodes.getNodeAs<UnaryOperator>("unary_op"))
    Loc = Op->getOperatorLoc();
  else if (const auto *Op =
               Result.Nodes.getNodeAs<CXXOperatorCallExpr>("overloaded_op"))
    Loc = Op->getOperatorLoc();

  if (Loc.isInvalid())
    return;

  Loc = Result.SourceManager->getSpellingLoc(Loc);
  if (Loc.isInvalid() || Loc.isMacroID())
    return;

  const CharSourceRange TokenRange = CharSourceRange::getTokenRange(Loc);
  if (TokenRange.isInvalid())
    return;

  StringRef Spelling = Lexer::getSourceText(TokenRange, *Result.SourceManager,
                                            Result.Context->getLangOpts());
  StringRef TranslatedSpelling = translate(Spelling);

  if (TranslatedSpelling.empty())
    return;

  std::string FixSpelling = TranslatedSpelling.str();

  StringRef SourceRepresentation = "an alternative";
  StringRef TargetRepresentation = "a traditional";
  if (needEscaping(TranslatedSpelling)) {
    SourceRepresentation = "a traditional";
    TargetRepresentation = "an alternative";

    StringRef SpellingEx = Lexer::getSourceText(
        CharSourceRange::getCharRange(
            TokenRange.getBegin().getLocWithOffset(-1),
            TokenRange.getBegin().getLocWithOffset(Spelling.size() + 1U)),
        *Result.SourceManager, Result.Context->getLangOpts());
    if (SpellingEx.empty() || !isSeparator(SpellingEx.front()))
      FixSpelling.insert(FixSpelling.begin(), ' ');
    if (SpellingEx.empty() || !isSeparator(SpellingEx.back()))
      FixSpelling.push_back(' ');
  }

  diag(
      Loc,
      "'%0' is %1 token spelling, consider using %2 token '%3' for consistency")
      << Spelling << SourceRepresentation << TargetRepresentation
      << TranslatedSpelling
      << FixItHint::CreateReplacement(TokenRange, FixSpelling);
}

} // namespace clang::tidy::readability
