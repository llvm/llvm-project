//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_LEXER_UTILS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_LEXER_UTILS_H

#include "clang/AST/ASTContext.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Lex/Lexer.h"
#include <optional>
#include <utility>

namespace clang {

class Stmt;

namespace tidy::utils::lexer {

/// Returns previous token or ``tok::unknown`` if not found.
Token getPreviousToken(SourceLocation Location, const SourceManager &SM,
                       const LangOptions &LangOpts, bool SkipComments = true);
std::pair<Token, SourceLocation>
getPreviousTokenAndStart(SourceLocation Location, const SourceManager &SM,
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
  if (Start.isInvalid() || Start.isMacroID())
    return {};
  while (true) {
    SourceLocation L = findPreviousTokenStart(Start, SM, LangOpts);
    if (L.isInvalid() || L.isMacroID())
      return {};

    Token T;
    // Returning 'true' is used to signal failure to retrieve the token.
    if (Lexer::getRawToken(L, T, SM, LangOpts, /*IgnoreWhiteSpace=*/true))
      return {};

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
    std::optional<Token> CurrentToken =
        Lexer::findNextToken(Start, SM, LangOpts);

    if (!CurrentToken)
      return {};

    Token PotentialMatch = *CurrentToken;
    if (PotentialMatch.isOneOf(TK, TKs...))
      return PotentialMatch.getLocation();

    // If we reach the end of the file, and eof is not the target token, we stop
    // the loop, otherwise we will get infinite loop (findNextToken will return
    // eof on eof).
    if (PotentialMatch.is(tok::eof))
      return {};
    Start = PotentialMatch.getLastLoc();
  }
}

inline std::optional<Token>
findNextTokenIncludingComments(SourceLocation Start, const SourceManager &SM,
                               const LangOptions &LangOpts) {
  return Lexer::findNextToken(Start, SM, LangOpts, true);
}

// Finds next token that's not a comment.
std::optional<Token> findNextTokenSkippingComments(SourceLocation Start,
                                                   const SourceManager &SM,
                                                   const LangOptions &LangOpts);

/// Re-lex the provide \p Range and return \c false if either a macro spans
/// multiple tokens, a pre-processor directive or failure to retrieve the
/// next token is found, otherwise \c true.
bool rangeContainsExpansionsOrDirectives(SourceRange Range,
                                         const SourceManager &SM,
                                         const LangOptions &LangOpts);

/// Assuming that ``Range`` spans a CVR-qualified type, returns the
/// token in ``Range`` that is responsible for the qualification. ``Range``
/// must be valid with respect to ``SM``.  Returns ``std::nullopt`` if no
/// qualifying tokens are found.
/// \note: doesn't support member function qualifiers.
std::optional<Token> getQualifyingToken(tok::TokenKind TK,
                                        CharSourceRange Range,
                                        const ASTContext &Context,
                                        const SourceManager &SM);

/// Stmt->getEndLoc does not always behave the same way depending on Token type.
/// See implementation for exceptions.
SourceLocation getUnifiedEndLoc(const Stmt &S, const SourceManager &SM,
                                const LangOptions &LangOpts);

/// For a given FunctionDecl returns the location where you would need to place
/// the noexcept specifier.
SourceLocation getLocationForNoexceptSpecifier(const FunctionDecl *FuncDecl,
                                               const SourceManager &SM);

} // namespace tidy::utils::lexer
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_LEXER_UTILS_H
