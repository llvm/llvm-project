//===--- StaticAssertCheck.cpp - clang-tidy -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "StaticAssertCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <optional>
#include <string>

using namespace clang::ast_matchers;

namespace clang::tidy::misc {

StaticAssertCheck::StaticAssertCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context) {}

void StaticAssertCheck::registerMatchers(MatchFinder *Finder) {
  auto NegatedString = unaryOperator(
      hasOperatorName("!"), hasUnaryOperand(ignoringImpCasts(stringLiteral())));
  auto IsAlwaysFalse =
      expr(anyOf(cxxBoolLiteral(equals(false)), integerLiteral(equals(0)),
                 cxxNullPtrLiteralExpr(), gnuNullExpr(), NegatedString))
          .bind("isAlwaysFalse");
  auto IsAlwaysFalseWithCast = ignoringParenImpCasts(anyOf(
      IsAlwaysFalse, cStyleCastExpr(has(ignoringParenImpCasts(IsAlwaysFalse)))
                         .bind("castExpr")));
  auto AssertExprRoot = anyOf(
      binaryOperator(
          hasAnyOperatorName("&&", "=="),
          hasEitherOperand(ignoringImpCasts(stringLiteral().bind("assertMSG"))),
          anyOf(binaryOperator(hasEitherOperand(IsAlwaysFalseWithCast)),
                anything()))
          .bind("assertExprRoot"),
      IsAlwaysFalse);
  auto NonConstexprFunctionCall =
      callExpr(hasDeclaration(functionDecl(unless(isConstexpr()))));
  auto AssertCondition =
      expr(
          anyOf(expr(ignoringParenCasts(anyOf(
                    AssertExprRoot, unaryOperator(hasUnaryOperand(
                                        ignoringParenCasts(AssertExprRoot)))))),
                anything()),
          unless(findAll(NonConstexprFunctionCall)))
          .bind("condition");
  auto Condition =
      anyOf(ignoringParenImpCasts(callExpr(
                hasDeclaration(functionDecl(hasName("__builtin_expect"))),
                hasArgument(0, AssertCondition))),
            AssertCondition);

  Finder->addMatcher(conditionalOperator(hasCondition(Condition),
                                         unless(isInTemplateInstantiation()))
                         .bind("condStmt"),
                     this);

  Finder->addMatcher(
      ifStmt(hasCondition(Condition), unless(isInTemplateInstantiation()))
          .bind("condStmt"),
      this);
}

void StaticAssertCheck::check(const MatchFinder::MatchResult &Result) {
  const ASTContext *ASTCtx = Result.Context;
  const LangOptions &Opts = ASTCtx->getLangOpts();
  const SourceManager &SM = ASTCtx->getSourceManager();
  const auto *CondStmt = Result.Nodes.getNodeAs<Stmt>("condStmt");
  const auto *Condition = Result.Nodes.getNodeAs<Expr>("condition");
  const auto *IsAlwaysFalse = Result.Nodes.getNodeAs<Expr>("isAlwaysFalse");
  const auto *AssertMSG = Result.Nodes.getNodeAs<StringLiteral>("assertMSG");
  const auto *AssertExprRoot =
      Result.Nodes.getNodeAs<BinaryOperator>("assertExprRoot");
  const auto *CastExpr = Result.Nodes.getNodeAs<CStyleCastExpr>("castExpr");
  SourceLocation AssertExpansionLoc = CondStmt->getBeginLoc();

  if (!AssertExpansionLoc.isValid() || !AssertExpansionLoc.isMacroID())
    return;

  StringRef MacroName =
      Lexer::getImmediateMacroName(AssertExpansionLoc, SM, Opts);

  if (MacroName != "assert" || Condition->isValueDependent() ||
      Condition->isTypeDependent() || Condition->isInstantiationDependent() ||
      !Condition->isEvaluatable(*ASTCtx))
    return;

  // False literal is not the result of macro expansion.
  if (IsAlwaysFalse && (!CastExpr || CastExpr->getType()->isPointerType())) {
    SourceLocation FalseLiteralLoc =
        SM.getImmediateSpellingLoc(IsAlwaysFalse->getExprLoc());
    if (!FalseLiteralLoc.isMacroID())
      return;

    StringRef FalseMacroName =
        Lexer::getImmediateMacroName(FalseLiteralLoc, SM, Opts);
    if (FalseMacroName.compare_insensitive("false") == 0 ||
        FalseMacroName.compare_insensitive("null") == 0)
      return;
  }

  SourceLocation AssertLoc = SM.getImmediateMacroCallerLoc(AssertExpansionLoc);

  SmallVector<FixItHint, 4> FixItHints;
  SourceLocation LastParenLoc;
  if (AssertLoc.isValid() && !AssertLoc.isMacroID() &&
      (LastParenLoc = getLastParenLoc(ASTCtx, AssertLoc)).isValid()) {
    FixItHints.push_back(
        FixItHint::CreateReplacement(SourceRange(AssertLoc), "static_assert"));

    if (AssertExprRoot) {
      FixItHints.push_back(FixItHint::CreateRemoval(
          SourceRange(AssertExprRoot->getOperatorLoc())));
      FixItHints.push_back(FixItHint::CreateRemoval(
          SourceRange(AssertMSG->getBeginLoc(), AssertMSG->getEndLoc())));
      FixItHints.push_back(FixItHint::CreateInsertion(
          LastParenLoc, (Twine(", \"") + AssertMSG->getString() + "\"").str()));
    } else if (!Opts.CPlusPlus17) {
      FixItHints.push_back(FixItHint::CreateInsertion(LastParenLoc, ", \"\""));
    }
  }

  diag(AssertLoc, "found assert() that could be replaced by static_assert()")
      << FixItHints;
}

SourceLocation StaticAssertCheck::getLastParenLoc(const ASTContext *ASTCtx,
                                                  SourceLocation AssertLoc) {
  const LangOptions &Opts = ASTCtx->getLangOpts();
  const SourceManager &SM = ASTCtx->getSourceManager();

  std::optional<llvm::MemoryBufferRef> Buffer =
      SM.getBufferOrNone(SM.getFileID(AssertLoc));
  if (!Buffer)
    return {};

  const char *BufferPos = SM.getCharacterData(AssertLoc);

  Token Token;
  Lexer Lexer(SM.getLocForStartOfFile(SM.getFileID(AssertLoc)), Opts,
              Buffer->getBufferStart(), BufferPos, Buffer->getBufferEnd());

  //        assert                          first left parenthesis
  if (Lexer.LexFromRawLexer(Token) || Lexer.LexFromRawLexer(Token) ||
      !Token.is(tok::l_paren))
    return {};

  unsigned int ParenCount = 1;
  while (ParenCount && !Lexer.LexFromRawLexer(Token)) {
    if (Token.is(tok::l_paren))
      ++ParenCount;
    else if (Token.is(tok::r_paren))
      --ParenCount;
  }

  return Token.getLocation();
}

} // namespace clang::tidy::misc
