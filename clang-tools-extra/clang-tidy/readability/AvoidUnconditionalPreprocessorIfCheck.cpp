//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AvoidUnconditionalPreprocessorIfCheck.h"
#include "../utils/LexerUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/STLExtras.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

namespace {
struct AvoidUnconditionalPreprocessorIfPPCallbacks : public PPCallbacks {
  explicit AvoidUnconditionalPreprocessorIfPPCallbacks(ClangTidyCheck &Check,
                                                       Preprocessor &PP)
      : Check(Check), PP(PP) {}

  void If(SourceLocation Loc, SourceRange ConditionRange,
          ConditionValueKind ConditionValue) override {
    if (ConditionValue == CVK_NotEvaluated)
      return;
    SourceManager &SM = PP.getSourceManager();
    if (!isImmutable(SM, PP.getLangOpts(), ConditionRange))
      return;

    if (ConditionValue == CVK_True)
      Check.diag(Loc, "preprocessor condition is always 'true', consider "
                      "removing condition but leaving its contents");
    else
      Check.diag(Loc, "preprocessor condition is always 'false', consider "
                      "removing both the condition and its contents");
  }

  bool isImmutable(SourceManager &SM, const LangOptions &LangOpts,
                   SourceRange ConditionRange) {
    const SourceLocation Loc = ConditionRange.getBegin();
    if (Loc.isMacroID())
      return false;

    const std::vector<Token> Tokens = utils::lexer::getRawTokens(
        CharSourceRange::getTokenRange(ConditionRange), SM, LangOpts);
    if (Tokens.empty())
      return false;
    return llvm::all_of(
        Tokens, [this](const Token &Tok) { return isImmutableToken(Tok); });
  }

  bool isImmutableToken(const Token &Tok) {
    switch (Tok.getKind()) {
    case tok::eod:
    case tok::eof:
    case tok::numeric_constant:
    case tok::char_constant:
    case tok::wide_char_constant:
    case tok::utf8_char_constant:
    case tok::utf16_char_constant:
    case tok::utf32_char_constant:
    case tok::string_literal:
    case tok::wide_string_literal:
    case tok::comment:
      return true;
    case tok::raw_identifier:
      return (Tok.getRawIdentifier() == "true" ||
              Tok.getRawIdentifier() == "false");
    default:
      return Tok.getKind() >= tok::l_square &&
             Tok.getKind() <= tok::greatergreatergreater;
    }
  }

  ClangTidyCheck &Check;
  Preprocessor &PP;
};

} // namespace

void AvoidUnconditionalPreprocessorIfCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  PP->addPPCallbacks(
      std::make_unique<AvoidUnconditionalPreprocessorIfPPCallbacks>(*this,
                                                                    *PP));
}

} // namespace clang::tidy::readability
