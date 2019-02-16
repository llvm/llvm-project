//===---------- ASTUtils.cpp - clang-tidy ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ASTUtils.h"

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"

namespace clang {
namespace tidy {
namespace utils {
using namespace ast_matchers;

const FunctionDecl *getSurroundingFunction(ASTContext &Context,
                                           const Stmt &Statement) {
  return selectFirst<const FunctionDecl>(
      "function", match(stmt(hasAncestor(functionDecl().bind("function"))),
                        Statement, Context));
}

bool IsBinaryOrTernary(const Expr *E) {
  const Expr *E_base = E->IgnoreImpCasts();
  if (clang::isa<clang::BinaryOperator>(E_base) ||
      clang::isa<clang::ConditionalOperator>(E_base)) {
    return true;
  }

  if (const auto *Operator =
          clang::dyn_cast<clang::CXXOperatorCallExpr>(E_base)) {
    return Operator->isInfixBinaryOp();
  }

  return false;
}

bool exprHasBitFlagWithSpelling(const Expr *Flags, const SourceManager &SM,
                                const LangOptions &LangOpts,
                                StringRef FlagName) {
  // If the Flag is an integer constant, check it.
  if (isa<IntegerLiteral>(Flags)) {
    if (!SM.isMacroBodyExpansion(Flags->getBeginLoc()) &&
        !SM.isMacroArgExpansion(Flags->getBeginLoc()))
      return false;

    // Get the macro name.
    auto MacroName = Lexer::getSourceText(
        CharSourceRange::getTokenRange(Flags->getSourceRange()), SM, LangOpts);

    return MacroName == FlagName;
  }
  // If it's a binary OR operation.
  if (const auto *BO = dyn_cast<BinaryOperator>(Flags))
    if (BO->getOpcode() == clang::BinaryOperatorKind::BO_Or)
      return exprHasBitFlagWithSpelling(BO->getLHS()->IgnoreParenCasts(), SM,
                                        LangOpts, FlagName) ||
             exprHasBitFlagWithSpelling(BO->getRHS()->IgnoreParenCasts(), SM,
                                        LangOpts, FlagName);

  // Otherwise, assume it has the flag.
  return true;
}

bool rangeIsEntirelyWithinMacroArgument(SourceRange Range,
                                        const SourceManager *SM) {
  // Check if the range is entirely contained within a macro argument.
  SourceLocation MacroArgExpansionStartForRangeBegin;
  SourceLocation MacroArgExpansionStartForRangeEnd;
  bool RangeIsEntirelyWithinMacroArgument =
      SM &&
      SM->isMacroArgExpansion(Range.getBegin(),
                              &MacroArgExpansionStartForRangeBegin) &&
      SM->isMacroArgExpansion(Range.getEnd(),
                              &MacroArgExpansionStartForRangeEnd) &&
      MacroArgExpansionStartForRangeBegin == MacroArgExpansionStartForRangeEnd;

  return RangeIsEntirelyWithinMacroArgument;
}

bool rangeContainsMacroExpansion(SourceRange Range, const SourceManager *SM) {
  return rangeIsEntirelyWithinMacroArgument(Range, SM) ||
         Range.getBegin().isMacroID() || Range.getEnd().isMacroID();
}

bool rangeCanBeFixed(SourceRange Range, const SourceManager *SM) {
  return utils::rangeIsEntirelyWithinMacroArgument(Range, SM) ||
         !utils::rangeContainsMacroExpansion(Range, SM);
}

} // namespace utils
} // namespace tidy
} // namespace clang
