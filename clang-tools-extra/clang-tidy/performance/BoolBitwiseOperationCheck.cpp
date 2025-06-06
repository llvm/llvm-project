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
#include <array>
#include <optional>
#include <utility>

using namespace clang::ast_matchers;

namespace clang::tidy::performance {

static std::string tryPrintVariable(const BinaryOperator *E) {
  if (E->isCompoundAssignmentOp()) {
    const auto *DelcRefLHS =
        dyn_cast<DeclRefExpr>(E->getLHS()->IgnoreImpCasts());
    if (DelcRefLHS)
      return "variable '" + DelcRefLHS->getDecl()->getNameAsString() + "'";
  }
  return "values";
}

static bool hasExplicitParentheses(const Expr *E, const SourceManager &SM,
                                   const LangOptions &LangOpts) {
  if (!E)
    return false;

  const SourceLocation Start = E->getBeginLoc();
  const SourceLocation End = E->getEndLoc();

  if (Start.isMacroID() || End.isMacroID() || !Start.isValid() ||
      !End.isValid())
    return false;

  const std::optional<Token> PrevTok =
      Lexer::findPreviousToken(Start, SM, LangOpts, /*IncludeComments=*/false);
  const std::optional<Token> NextTok =
      Lexer::findNextToken(End, SM, LangOpts, /*IncludeComments=*/false);

  return (PrevTok && PrevTok->is(tok::l_paren)) &&
         (NextTok && NextTok->is(tok::r_paren));
}

template <typename AstNode>
static bool isInTemplateFunction(const AstNode *AN, ASTContext &Context) {
  DynTypedNodeList Parents = Context.getParents(*AN);
  for (const auto &Parent : Parents) {
    if (const auto *FD = Parent.template get<FunctionDecl>())
      return FD->isTemplateInstantiation() ||
             FD->getTemplatedKind() != FunctionDecl::TK_NonTemplate;
    if (const auto *S = Parent.template get<Stmt>())
      return isInTemplateFunction(S, Context);
  }
  return false;
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

static llvm::StringRef translate(llvm::StringRef Value) {
  for (const auto &[Bitwise, Logical] : OperatorsTransformation) {
    if (Value == Bitwise)
      return Logical;
  }

  return {};
}

BoolBitwiseOperationCheck::BoolBitwiseOperationCheck(StringRef Name,
                                                     ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      StrictMode(Options.get("StrictMode", true)) {}

void BoolBitwiseOperationCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "StrictMode", StrictMode);
}

void BoolBitwiseOperationCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      binaryOperator(
          unless(isExpansionInSystemHeader()),
          hasAnyOperatorName("|", "&", "|=", "&="),
          hasEitherOperand(expr(ignoringImpCasts(hasType(booleanType())))),
          optionally(hasAncestor( // to simple implement transformations like
                                  // `a&&b|c` -> `a&&(b||c)`
              binaryOperator().bind("p"))))
          .bind("op"),
      this);
}

void BoolBitwiseOperationCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedExpr = Result.Nodes.getNodeAs<BinaryOperator>("op");

  auto Diag = diag(MatchedExpr->getOperatorLoc(),
                   "use logical operator '%0' for boolean %1 instead of "
                   "bitwise operator '%2'")
              << translate(MatchedExpr->getOpcodeStr())
              << tryPrintVariable(MatchedExpr) << MatchedExpr->getOpcodeStr();

  if (isInTemplateFunction(MatchedExpr, *Result.Context))
    return;

  const bool HasVolatileOperand = llvm::any_of(
      std::array{MatchedExpr->getLHS(), MatchedExpr->getRHS()},
      [](const Expr *E) {
        return E->IgnoreImpCasts()->getType().isVolatileQualified();
      });
  const bool HasSideEffects =
      MatchedExpr->getRHS()->HasSideEffects(*Result.Context, StrictMode);
  if (HasVolatileOperand || HasSideEffects)
    return;

  SourceLocation Loc = MatchedExpr->getOperatorLoc();

  if (Loc.isInvalid() || Loc.isMacroID())
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

  const std::string FixSpelling = TranslatedSpelling.str();

  FixItHint InsertEqual, ReplaceOperator, InsertBrace1, InsertBrace2;
  if (MatchedExpr->isCompoundAssignmentOp()) {
    const auto *DelcRefLHS =
        dyn_cast<DeclRefExpr>(MatchedExpr->getLHS()->IgnoreImpCasts());
    if (!DelcRefLHS)
      return;
    const SourceLocation LocLHS = DelcRefLHS->getEndLoc();
    if (LocLHS.isInvalid() || LocLHS.isMacroID())
      return;
    const SourceLocation InsertLoc = clang::Lexer::getLocForEndOfToken(
        LocLHS, 0, *Result.SourceManager, Result.Context->getLangOpts());
    if (InsertLoc.isInvalid() || InsertLoc.isMacroID()) {
      return;
    }
    InsertEqual = FixItHint::CreateInsertion(
        InsertLoc, " = " + DelcRefLHS->getDecl()->getNameAsString());
  }
  ReplaceOperator = FixItHint::CreateReplacement(TokenRange, FixSpelling);

  std::optional<BinaryOperatorKind> ParentOpcode;
  if (const auto *Parent = Result.Nodes.getNodeAs<BinaryOperator>("p"); Parent)
    ParentOpcode = Parent->getOpcode();

  const auto *RHS =
      dyn_cast<BinaryOperator>(MatchedExpr->getRHS()->IgnoreParenCasts());
  std::optional<BinaryOperatorKind> RHSOpcode;
  if (RHS)
    RHSOpcode = RHS->getOpcode();

  const BinaryOperator *SurroundedExpr = nullptr;
  if ((MatchedExpr->getOpcode() == BO_Or && ParentOpcode.has_value() &&
       *ParentOpcode == BO_LAnd) ||
      (MatchedExpr->getOpcode() == BO_And && ParentOpcode.has_value() &&
       llvm::is_contained({BO_Xor, BO_Or}, *ParentOpcode)))
    SurroundedExpr = MatchedExpr;
  else if (MatchedExpr->getOpcode() == BO_AndAssign && RHSOpcode.has_value() &&
           *RHSOpcode == BO_LOr)
    SurroundedExpr = RHS;

  if (hasExplicitParentheses(SurroundedExpr, *Result.SourceManager,
                             Result.Context->getLangOpts()))
    SurroundedExpr = nullptr;

  if (SurroundedExpr) {
    const SourceLocation InsertFirstLoc = SurroundedExpr->getBeginLoc();
    const SourceLocation InsertSecondLoc = clang::Lexer::getLocForEndOfToken(
        SurroundedExpr->getEndLoc(), 0, *Result.SourceManager,
        Result.Context->getLangOpts());
    if (InsertFirstLoc.isInvalid() || InsertFirstLoc.isMacroID() ||
        InsertSecondLoc.isInvalid() || InsertSecondLoc.isMacroID())
      return;
    InsertBrace1 = FixItHint::CreateInsertion(InsertFirstLoc, "(");
    InsertBrace2 = FixItHint::CreateInsertion(InsertSecondLoc, ")");
  }

  Diag << InsertEqual << ReplaceOperator << InsertBrace1 << InsertBrace2;
}

} // namespace clang::tidy::performance
