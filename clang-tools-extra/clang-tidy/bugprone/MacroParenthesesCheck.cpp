//===--- MacroParenthesesCheck.cpp - clang-tidy----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MacroParenthesesCheck.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"

namespace clang {
namespace tidy {
namespace bugprone {

namespace {
class MacroParenthesesPPCallbacks : public PPCallbacks {
public:
  MacroParenthesesPPCallbacks(Preprocessor *PP, MacroParenthesesCheck *Check)
      : PP(PP), Check(Check) {}

  void MacroDefined(const Token &MacroNameTok,
                    const MacroDirective *MD) override {
    replacementList(MacroNameTok, MD->getMacroInfo());
    argument(MacroNameTok, MD->getMacroInfo());
  }

private:
  /// Replacement list with calculations should be enclosed in parentheses.
  void replacementList(const Token &MacroNameTok, const MacroInfo *MI);

  /// Arguments should be enclosed in parentheses.
  void argument(const Token &MacroNameTok, const MacroInfo *MI);

  Preprocessor *PP;
  MacroParenthesesCheck *Check;
};
} // namespace

/// Is argument surrounded properly with parentheses/braces/squares/commas?
static bool isSurroundedLeft(const Token &T) {
  return T.isOneOf(tok::l_paren, tok::l_brace, tok::l_square, tok::comma,
                   tok::semi);
}

/// Is argument surrounded properly with parentheses/braces/squares/commas?
static bool isSurroundedRight(const Token &T) {
  return T.isOneOf(tok::r_paren, tok::r_brace, tok::r_square, tok::comma,
                   tok::semi);
}

/// Is given TokenKind a keyword?
static bool isKeyword(const Token &T) {
  // FIXME: better matching of keywords to avoid false positives.
  return T.isOneOf(tok::kw_if, tok::kw_case, tok::kw_const, tok::kw_struct);
}

/// Warning is written when one of these operators are not within parentheses.
static bool isWarnOp(const Token &T) {
  // FIXME: This is an initial list of operators. It can be tweaked later to
  // get more positives or perhaps avoid some false positive.
  return T.isOneOf(tok::plus, tok::minus, tok::star, tok::slash, tok::percent,
                   tok::amp, tok::pipe, tok::caret);
}

/// Is given Token a keyword that is used in variable declarations?
static bool isVarDeclKeyword(const Token &T) {
  return T.isOneOf(tok::kw_bool, tok::kw_char, tok::kw_short, tok::kw_int,
                   tok::kw_long, tok::kw_float, tok::kw_double, tok::kw_const,
                   tok::kw_enum, tok::kw_inline, tok::kw_static, tok::kw_struct,
                   tok::kw_signed, tok::kw_unsigned);
}

/// Is there a possible variable declaration at Tok?
static bool possibleVarDecl(const MacroInfo *MI, const Token *Tok) {
  if (Tok == MI->tokens_end())
    return false;

  // If we see int/short/struct/etc., just assume this is a variable
  // declaration.
  if (isVarDeclKeyword(*Tok))
    return true;

  // Variable declarations start with identifier or coloncolon.
  if (!Tok->isOneOf(tok::identifier, tok::raw_identifier, tok::coloncolon))
    return false;

  // Skip possible types, etc
  while (Tok != MI->tokens_end() &&
         Tok->isOneOf(tok::identifier, tok::raw_identifier, tok::coloncolon,
                      tok::star, tok::amp, tok::ampamp, tok::less,
                      tok::greater))
    Tok++;

  // Return true for possible variable declarations.
  return Tok == MI->tokens_end() ||
         Tok->isOneOf(tok::equal, tok::semi, tok::l_square, tok::l_paren) ||
         isVarDeclKeyword(*Tok);
}

void MacroParenthesesPPCallbacks::replacementList(const Token &MacroNameTok,
                                                  const MacroInfo *MI) {
  // Make sure macro replacement isn't a variable declaration.
  if (possibleVarDecl(MI, MI->tokens_begin()))
    return;

  // Count how deep we are in parentheses/braces/squares.
  int Count = 0;

  // SourceLocation for error
  SourceLocation Loc;

  for (auto TI = MI->tokens_begin(), TE = MI->tokens_end(); TI != TE; ++TI) {
    const Token &Tok = *TI;
    // Replacement list contains keywords, don't warn about it.
    if (isKeyword(Tok))
      return;
    // When replacement list contains comma/semi don't warn about it.
    if (Count == 0 && Tok.isOneOf(tok::comma, tok::semi))
      return;
    if (Tok.isOneOf(tok::l_paren, tok::l_brace, tok::l_square)) {
      ++Count;
    } else if (Tok.isOneOf(tok::r_paren, tok::r_brace, tok::r_square)) {
      --Count;
      // If there are unbalanced parentheses don't write any warning
      if (Count < 0)
        return;
    } else if (Count == 0 && isWarnOp(Tok)) {
      // Heuristic for macros that are clearly not intended to be enclosed in
      // parentheses, macro starts with operator. For example:
      // #define X     *10
      if (TI == MI->tokens_begin() && (TI + 1) != TE &&
          !Tok.isOneOf(tok::plus, tok::minus))
        return;
      // Don't warn about this macro if the last token is a star. For example:
      // #define X    void *
      if ((TE - 1)->is(tok::star))
        return;

      Loc = Tok.getLocation();
    }
  }
  if (Loc.isValid()) {
    const Token &Last = *(MI->tokens_end() - 1);
    Check->diag(Loc, "macro replacement list should be enclosed in parentheses")
        << FixItHint::CreateInsertion(MI->tokens_begin()->getLocation(), "(")
        << FixItHint::CreateInsertion(Last.getLocation().getLocWithOffset(
                                          PP->getSpelling(Last).length()),
                                      ")");
  }
}

void MacroParenthesesPPCallbacks::argument(const Token &MacroNameTok,
                                           const MacroInfo *MI) {

  // Skip variable declaration.
  bool VarDecl = possibleVarDecl(MI, MI->tokens_begin());

  for (auto TI = MI->tokens_begin(), TE = MI->tokens_end(); TI != TE; ++TI) {
    // First token.
    if (TI == MI->tokens_begin())
      continue;

    // Last token.
    if ((TI + 1) == MI->tokens_end())
      continue;

    const Token &Prev = *(TI - 1);
    const Token &Next = *(TI + 1);

    const Token &Tok = *TI;

    // There should not be extra parentheses in possible variable declaration.
    if (VarDecl) {
      if (Tok.isOneOf(tok::equal, tok::semi, tok::l_square, tok::l_paren))
        VarDecl = false;
      continue;
    }

    // Only interested in identifiers.
    if (!Tok.isOneOf(tok::identifier, tok::raw_identifier))
      continue;

    // Only interested in macro arguments.
    if (MI->getParameterNum(Tok.getIdentifierInfo()) < 0)
      continue;

    // Argument is surrounded with parentheses/squares/braces/commas.
    if (isSurroundedLeft(Prev) && isSurroundedRight(Next))
      continue;

    // Don't warn after hash/hashhash or before hashhash.
    if (Prev.isOneOf(tok::hash, tok::hashhash) || Next.is(tok::hashhash))
      continue;

    // Argument is a struct member.
    if (Prev.isOneOf(tok::period, tok::arrow, tok::coloncolon, tok::arrowstar,
                     tok::periodstar))
      continue;

    // Argument is a namespace or class.
    if (Next.is(tok::coloncolon))
      continue;

    // String concatenation.
    if (isStringLiteral(Prev.getKind()) || isStringLiteral(Next.getKind()))
      continue;

    // Type/Var.
    if (isAnyIdentifier(Prev.getKind()) || isKeyword(Prev) ||
        isAnyIdentifier(Next.getKind()) || isKeyword(Next))
      continue;

    // Initialization.
    if (Next.is(tok::l_paren))
      continue;

    // Cast.
    if (Prev.is(tok::l_paren) && Next.is(tok::star) &&
        TI + 2 != MI->tokens_end() && (TI + 2)->is(tok::r_paren))
      continue;

    // Assignment/return, i.e. '=x;' or 'return x;'.
    if (Prev.isOneOf(tok::equal, tok::kw_return) && Next.is(tok::semi))
      continue;

    // C++ template parameters.
    if (PP->getLangOpts().CPlusPlus && Prev.isOneOf(tok::comma, tok::less) &&
        Next.isOneOf(tok::comma, tok::greater))
      continue;

    // Namespaces.
    if (Prev.is(tok::kw_namespace))
      continue;

    // Variadic templates
    if (MI->isVariadic())
      continue;

    Check->diag(Tok.getLocation(), "macro argument should be enclosed in "
                                   "parentheses")
        << FixItHint::CreateInsertion(Tok.getLocation(), "(")
        << FixItHint::CreateInsertion(Tok.getLocation().getLocWithOffset(
                                          PP->getSpelling(Tok).length()),
                                      ")");
  }
}

void MacroParenthesesCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  PP->addPPCallbacks(std::make_unique<MacroParenthesesPPCallbacks>(PP, this));
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
