//===--- LexerUtils.h - clang-tidy-------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_LEXER_UTILS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_LEXER_UTILS_H

#include "clang/AST/ASTContext.h"
#include "clang/Lex/Lexer.h"

namespace clang {
namespace tidy {
namespace utils {
namespace lexer {

/// Returns previous token or ``tok::unknown`` if not found.
Token getPreviousToken(SourceLocation Location, const SourceManager &SM,
                       const LangOptions &LangOpts, bool SkipComments = true);

SourceLocation findPreviousTokenStart(SourceLocation Start,
                                      const SourceManager &SM,
                                      const LangOptions &LangOpts);

SourceLocation findPreviousTokenKind(SourceLocation Start,
                                     const SourceManager &SM,
                                     const LangOptions &LangOpts,
                                     tok::TokenKind TK);

SourceLocation findNextTerminator(SourceLocation Start, const SourceManager &SM,
                                  const LangOptions &LangOpts);

template <typename TokenKind, typename... TokenKinds>
SourceLocation findPreviousAnyTokenKind(SourceLocation Start,
                                        const SourceManager &SM,
                                        const LangOptions &LangOpts,
                                        TokenKind TK, TokenKinds... TKs) {
  while (true) {
    SourceLocation L = findPreviousTokenStart(Start, SM, LangOpts);
    if (L.isInvalid() || L.isMacroID())
      return SourceLocation();

    Token T;
    // Returning 'true' is used to signal failure to retrieve the token.
    if (Lexer::getRawToken(L, T, SM, LangOpts))
      return SourceLocation();

    if (T.isOneOf(TK, TKs...))
      return T.getLocation();

    Start = L;
  }
}

template <typename TokenKind, typename... TokenKinds>
SourceLocation findNextAnyTokenKind(SourceLocation Start,
                                    const SourceManager &SM,
                                    const LangOptions &LangOpts, TokenKind TK,
                                    TokenKinds... TKs) {
  while (true) {
    Optional<Token> CurrentToken = Lexer::findNextToken(Start, SM, LangOpts);

    if (!CurrentToken)
      return SourceLocation();

    Token PotentialMatch = *CurrentToken;
    if (PotentialMatch.isOneOf(TK, TKs...))
      return PotentialMatch.getLocation();

    Start = PotentialMatch.getLastLoc();
  }
}

/// Re-lex the provide \p Range and return \c false if either a macro spans
/// multiple tokens, a pre-processor directive or failure to retrieve the
/// next token is found, otherwise \c true.
bool rangeContainsExpansionsOrDirectives(SourceRange Range,
                                         const SourceManager &SM,
                                         const LangOptions &LangOpts);

/// Assuming that ``Range`` spans a const-qualified type, returns the ``const``
/// token in ``Range`` that is responsible for const qualification. ``Range``
/// must be valid with respect to ``SM``.  Returns ``None`` if no ``const``
/// tokens are found.
llvm::Optional<Token> getConstQualifyingToken(CharSourceRange Range,
                                              const ASTContext &Context,
                                              const SourceManager &SM);

} // namespace lexer
} // namespace utils
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_LEXER_UTILS_H
