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

using namespace clang::ast_matchers;

namespace clang::tidy::performance {
namespace {
bool isFullyInsideMacro(const BinaryOperator *BO, const SourceManager &SM) {
  SourceLocation Begin = BO->getBeginLoc();
  SourceLocation End = BO->getEndLoc();
  
  // Если хотя бы одна часть оператора не в макросе — считаем его "не макросным"
  if (!Begin.isMacroID() || !End.isMacroID()) {
      return false;
  }
  
  // Проверяем, что начало и конец находятся в одном макросе
  return SM.getImmediateMacroCallerLoc(Begin) == SM.getImmediateMacroCallerLoc(End);
}

std::string getSpellingOpcode(const BinaryOperator &Expr, const SourceManager &SM,
                                                          const clang::LangOptions &LO) {
  SourceLocation Loc = Expr.getOperatorLoc();
  if (Loc.isValid()) {
    // TODO: bear it in mind
    // SourceLocation expansionLoc = Result.SourceManager->getExpansionLoc(Loc);
    // if (expansionLoc.isValid()) {
      Loc = SM.getSpellingLoc(Loc);
      if (Loc.isValid() && !Loc.isMacroID()) {
        const CharSourceRange TokenRange = CharSourceRange::getTokenRange(Loc);
        if (TokenRange.isValid()) {
          return Lexer::getSourceText(TokenRange, SM, LO).str();
        }
      }
    // }
  }
  return "";
}

std::string changeOpcode(llvm::StringRef Spelling) {
  if (Spelling == "|" || Spelling == "|=")
      return "||";
  else if (Spelling == "&" || Spelling == "&=")
      return "&&";
  else if (Spelling == "bitand" || Spelling == "and_eq")
    return "and";
  else if (Spelling == "bitor" || Spelling == "or_eq")
    return "or";
  return Spelling.str();
}
}

void BoolBitwiseOperationCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(binaryOperator(
        unless(isExpansionInSystemHeader()),
        hasAnyOperatorName("|", "&", "|=", "&="),
        hasEitherOperand(expr(ignoringImpCasts(hasType(booleanType())))),
        optionally(hasEitherOperand(
            expr(ignoringImpCasts(hasType(isVolatileQualified())))
                .bind("vol"))),
        optionally(hasAncestor(
            binaryOperator().bind("p"))),
        optionally(hasRHS(ignoringParenCasts(
            binaryOperator().bind("r")))),
        optionally(hasLHS(ignoringParenCasts(
            declRefExpr().bind("l")
        )))
        )
    .bind("op"), this);
}

void BoolBitwiseOperationCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedExpr = Result.Nodes.getNodeAs<BinaryOperator>("op");
  const SourceManager &SM = *Result.SourceManager;
  const clang::LangOptions &LO = Result.Context->getLangOpts();

  auto Diag = diag(MatchedExpr->getOperatorLoc(), "use logical operator instead of bitwise one for bool");

  const auto *VolatileOperand = Result.Nodes.getNodeAs<Expr>("vol");
  if (VolatileOperand || isFullyInsideMacro(MatchedExpr, SM))
    return;

  SourceLocation Loc = MatchedExpr->getOperatorLoc();
  if (Loc.isValid()) {
    // TODO: bear it in mind
    //SourceLocation expansionLoc = Result.SourceManager->getExpansionLoc(Loc);
    SourceLocation expansionLoc = Loc;
    if (expansionLoc.isValid()) {
      expansionLoc = SM.getFileLoc(expansionLoc);
      if (expansionLoc.isValid() && !expansionLoc.isMacroID()) {
        const CharSourceRange TokenRange = CharSourceRange::getTokenRange(expansionLoc);
        if (TokenRange.isValid()) {
          const std::string SpellingOpc = getSpellingOpcode(*MatchedExpr, SM, LO);
          if (SpellingOpc == "&=" || SpellingOpc == "|=" || SpellingOpc == "and_eq" || SpellingOpc == "or_eq") {
            const auto *DelcRefLHS = Result.Nodes.getNodeAs<DeclRefExpr>("l");
            if (!DelcRefLHS)
              return;
            const clang::SourceLocation EndLoc = // TODO: naming
                clang::Lexer::getLocForEndOfToken(DelcRefLHS->getEndLoc(), 0,
                                                  SM, LO);
            if (EndLoc.isInvalid()) {
              return;
            }
            Diag << FixItHint::CreateInsertion(EndLoc,
                                               " = " + DelcRefLHS->getDecl()->getNameAsString());
          }
          Diag << FixItHint::CreateReplacement(TokenRange, changeOpcode(SpellingOpc));
          const auto *Parent = Result.Nodes.getNodeAs<BinaryOperator>("p");
          const auto *RHS = Result.Nodes.getNodeAs<BinaryOperator>("r");
          const std::string ParentSpellingOpc = Parent ? getSpellingOpcode(*Parent, SM, LO) : "";
          const std::string RightSpellingOpc = RHS ? getSpellingOpcode(*RHS, SM, LO) : "";
          if (((SpellingOpc == "|" || SpellingOpc == "bitor") && (ParentSpellingOpc == "&&" || ParentSpellingOpc == "and")) ||
              ((SpellingOpc == "&" || SpellingOpc == "bitand") && (ParentSpellingOpc == "^" || ParentSpellingOpc == "xor"))) {
            const clang::SourceLocation StartLoc = MatchedExpr->getBeginLoc();
            const clang::SourceLocation EndLoc = // TODO: check for valid
                clang::Lexer::getLocForEndOfToken(MatchedExpr->getEndLoc(), 0, SM, LO);
            Diag << FixItHint::CreateInsertion(StartLoc, "(")
                 << FixItHint::CreateInsertion(EndLoc, ")");
          } else if ((SpellingOpc == "&=" || SpellingOpc == "and_eq") && (RightSpellingOpc == "||" || RightSpellingOpc == "or")) {
            const clang::SourceLocation StartLoc = RHS->getBeginLoc();
            const clang::SourceLocation EndLoc = // TODO: check for valid
                clang::Lexer::getLocForEndOfToken(RHS->getEndLoc(), 0, SM, LO);
            Diag << FixItHint::CreateInsertion(StartLoc, "(")
                 << FixItHint::CreateInsertion(EndLoc, ")");
          }
        }
      }
    }
  }
}

} // namespace clang::tidy::performance
