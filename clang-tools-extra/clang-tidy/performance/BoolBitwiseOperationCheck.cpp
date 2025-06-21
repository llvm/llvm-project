//===--- BoolBitwiseOperationCheck.cpp - clang-tidy -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BoolBitwiseOperationCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/ScopeExit.h"
#include <array>
#include <optional>
#include <utility>

using namespace clang::ast_matchers;

namespace clang::tidy::performance {

static const NamedDecl *
getLHSNamedDeclIfCompoundAssign(const BinaryOperator *BO) {
  if (BO->isCompoundAssignmentOp()) {
    const auto *DelcRefLHS =
        dyn_cast<DeclRefExpr>(BO->getLHS()->IgnoreImpCasts());
    return DelcRefLHS ? DelcRefLHS->getDecl() : nullptr;
  }
  return nullptr;
}

static bool hasExplicitParentheses(const Expr *E, const SourceManager &SM,
                                   const LangOptions &LangOpts) {
  if (!E)
    return false;

  const SourceLocation Start = E->getBeginLoc();
  const SourceLocation End = E->getEndLoc();

  if (Start.isMacroID() || End.isMacroID() || Start.isInvalid() ||
      End.isInvalid())
    return false;

  const std::optional<Token> PrevTok =
      Lexer::findPreviousToken(Start, SM, LangOpts, /*IncludeComments=*/false);
  const std::optional<Token> NextTok =
      Lexer::findNextToken(End, SM, LangOpts, /*IncludeComments=*/false);

  return (PrevTok && PrevTok->is(tok::l_paren)) &&
         (NextTok && NextTok->is(tok::r_paren));
}

constexpr std::array<std::pair<llvm::StringRef, llvm::StringRef>, 8U>
    OperatorsTransformation{{{"|", "||"},
                             {"|=", "||"},
                             {"&", "&&"},
                             {"&=", "&&"},
                             {"bitand", "and"},
                             {"and_eq", "and"},
                             {"bitor", "or"},
                             {"or_eq", "or"}}};

static std::string translate(llvm::StringRef Value) {
  for (const auto &[Bitwise, Logical] : OperatorsTransformation) {
    if (Value == Bitwise)
      return Logical.str();
  }

  return {};
}

BoolBitwiseOperationCheck::BoolBitwiseOperationCheck(StringRef Name,
                                                     ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      StrictMode(Options.get("StrictMode", true)),
      IgnoreMacros(Options.get("IgnoreMacros", false)) {}

void BoolBitwiseOperationCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "StrictMode", StrictMode);
  Options.store(Opts, "IgnoreMacros", IgnoreMacros);
}

void BoolBitwiseOperationCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      binaryOperator(
          unless(isExpansionInSystemHeader()),
          hasAnyOperatorName("|", "&", "|=", "&="),
          hasEitherOperand(expr(hasType(booleanType()))),
          optionally(hasParent( // to simple implement transformations like
                                // `a&&b|c` -> `a&&(b||c)`
              binaryOperator().bind("p"))))
          .bind("op"),
      this);
}

void BoolBitwiseOperationCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedExpr = Result.Nodes.getNodeAs<BinaryOperator>("op");

  auto DiagEmitterProc = [MatchedExpr, this] {
    const NamedDecl *ND = getLHSNamedDeclIfCompoundAssign(MatchedExpr);
    return diag(MatchedExpr->getOperatorLoc(),
                "use logical operator '%0' for boolean %select{variable "
                "%2|values}1 instead of "
                "bitwise operator '%3'")
           << translate(MatchedExpr->getOpcodeStr()) << (ND == nullptr) << ND
           << MatchedExpr->getOpcodeStr();
  };
  auto DiagEmitter = llvm::make_scope_exit(DiagEmitterProc);

  const bool HasVolatileOperand = llvm::any_of(
      std::array{MatchedExpr->getLHS(), MatchedExpr->getRHS()},
      [](const Expr *E) {
        return E->IgnoreImpCasts()->getType().isVolatileQualified();
      });
  if (HasVolatileOperand)
    return;

  const bool HasSideEffects =
      MatchedExpr->getRHS()->HasSideEffects(*Result.Context, StrictMode);
  if (HasSideEffects)
    return;

  SourceLocation Loc = MatchedExpr->getOperatorLoc();

  if (Loc.isInvalid() || Loc.isMacroID()) {
    if (IgnoreMacros)
      DiagEmitter.release();
    return;
  }

  Loc = Result.SourceManager->getSpellingLoc(Loc);
  if (Loc.isInvalid() || Loc.isMacroID()) {
    if (IgnoreMacros)
      DiagEmitter.release();
    return;
  }

  const CharSourceRange TokenRange = CharSourceRange::getTokenRange(Loc);
  if (TokenRange.isInvalid()) {
    if (IgnoreMacros)
      DiagEmitter.release();
    return;
  }

  StringRef Spelling = Lexer::getSourceText(TokenRange, *Result.SourceManager,
                                            Result.Context->getLangOpts());
  const std::string FixSpelling = translate(Spelling);

  if (FixSpelling.empty())
    return;

  FixItHint InsertEqual;
  if (MatchedExpr->isCompoundAssignmentOp()) {
    const auto *DelcRefLHS =
        dyn_cast<DeclRefExpr>(MatchedExpr->getLHS()->IgnoreImpCasts());
    if (!DelcRefLHS)
      return;
    const SourceLocation LocLHS = DelcRefLHS->getEndLoc();
    if (LocLHS.isInvalid() || LocLHS.isMacroID()) {
      if (IgnoreMacros)
        DiagEmitter.release();
      return;
    }
    const SourceLocation InsertLoc = clang::Lexer::getLocForEndOfToken(
        LocLHS, 0, *Result.SourceManager, Result.Context->getLangOpts());
    if (InsertLoc.isInvalid() || InsertLoc.isMacroID()) {
      if (IgnoreMacros)
        DiagEmitter.release();
      return;
    }
    InsertEqual = FixItHint::CreateInsertion(
        InsertLoc, " = " + DelcRefLHS->getDecl()->getNameAsString());
  }

  auto ReplaceOperator = FixItHint::CreateReplacement(TokenRange, FixSpelling);

  std::optional<BinaryOperatorKind> ParentOpcode;
  if (const auto *Parent = Result.Nodes.getNodeAs<BinaryOperator>("p"); Parent)
    ParentOpcode = Parent->getOpcode();

  const auto *RHS =
      dyn_cast<BinaryOperator>(MatchedExpr->getRHS()->IgnoreImpCasts());
  std::optional<BinaryOperatorKind> RHSOpcode;
  if (RHS)
    RHSOpcode = RHS->getOpcode();

  const BinaryOperator *SurroundedExpr = nullptr;
  if ((MatchedExpr->getOpcode() == BO_Or && ParentOpcode == BO_LAnd) ||
      (MatchedExpr->getOpcode() == BO_And &&
       llvm::is_contained({BO_Xor, BO_Or}, ParentOpcode)))
    SurroundedExpr = MatchedExpr;
  else if (MatchedExpr->getOpcode() == BO_AndAssign && RHSOpcode == BO_LOr)
    SurroundedExpr = RHS;

  if (hasExplicitParentheses(SurroundedExpr, *Result.SourceManager,
                             Result.Context->getLangOpts()))
    SurroundedExpr = nullptr;

  FixItHint InsertBrace1, InsertBrace2;
  if (SurroundedExpr) {
    const SourceLocation InsertFirstLoc = SurroundedExpr->getBeginLoc();
    const SourceLocation InsertSecondLoc = clang::Lexer::getLocForEndOfToken(
        SurroundedExpr->getEndLoc(), 0, *Result.SourceManager,
        Result.Context->getLangOpts());
    if (InsertFirstLoc.isInvalid() || InsertFirstLoc.isMacroID() ||
        InsertSecondLoc.isInvalid() || InsertSecondLoc.isMacroID()) {
      if (IgnoreMacros)
        DiagEmitter.release();
      return;
    }
    InsertBrace1 = FixItHint::CreateInsertion(InsertFirstLoc, "(");
    InsertBrace2 = FixItHint::CreateInsertion(InsertSecondLoc, ")");
  }

  DiagEmitter.release();
  DiagEmitterProc() << InsertEqual << ReplaceOperator << InsertBrace1
                    << InsertBrace2;
}

} // namespace clang::tidy::performance
