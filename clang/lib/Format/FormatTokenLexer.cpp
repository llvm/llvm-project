//===--- FormatTokenLexer.cpp - Lex FormatTokens -------------*- C++ ----*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements FormatTokenLexer, which tokenizes a source file
/// into a FormatToken stream suitable for ClangFormat.
///
//===----------------------------------------------------------------------===//

#include "FormatTokenLexer.h"
#include "FormatToken.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Format/Format.h"
#include "llvm/Support/Regex.h"

namespace clang {
namespace format {

FormatTokenLexer::FormatTokenLexer(
    const SourceManager &SourceMgr, FileID ID, unsigned Column,
    const FormatStyle &Style, encoding::Encoding Encoding,
    llvm::SpecificBumpPtrAllocator<FormatToken> &Allocator,
    IdentifierTable &IdentTable)
    : FormatTok(nullptr), IsFirstToken(true), StateStack({LexerState::NORMAL}),
      Column(Column), TrailingWhitespace(0), SourceMgr(SourceMgr), ID(ID),
      Style(Style), IdentTable(IdentTable), Keywords(IdentTable),
      Encoding(Encoding), Allocator(Allocator), FirstInLineIndex(0),
      FormattingDisabled(false), MacroBlockBeginRegex(Style.MacroBlockBegin),
      MacroBlockEndRegex(Style.MacroBlockEnd) {
  Lex.reset(new Lexer(ID, SourceMgr.getBufferOrFake(ID), SourceMgr,
                      getFormattingLangOpts(Style)));
  Lex->SetKeepWhitespaceMode(true);

  for (const std::string &ForEachMacro : Style.ForEachMacros)
    Macros.insert({&IdentTable.get(ForEachMacro), TT_ForEachMacro});
  for (const std::string &IfMacro : Style.IfMacros)
    Macros.insert({&IdentTable.get(IfMacro), TT_IfMacro});
  for (const std::string &AttributeMacro : Style.AttributeMacros)
    Macros.insert({&IdentTable.get(AttributeMacro), TT_AttributeMacro});
  for (const std::string &StatementMacro : Style.StatementMacros)
    Macros.insert({&IdentTable.get(StatementMacro), TT_StatementMacro});
  for (const std::string &TypenameMacro : Style.TypenameMacros)
    Macros.insert({&IdentTable.get(TypenameMacro), TT_TypenameMacro});
  for (const std::string &NamespaceMacro : Style.NamespaceMacros)
    Macros.insert({&IdentTable.get(NamespaceMacro), TT_NamespaceMacro});
  for (const std::string &WhitespaceSensitiveMacro :
       Style.WhitespaceSensitiveMacros) {
    Macros.insert(
        {&IdentTable.get(WhitespaceSensitiveMacro), TT_UntouchableMacroFunc});
  }
  for (const std::string &StatementAttributeLikeMacro :
       Style.StatementAttributeLikeMacros)
    Macros.insert({&IdentTable.get(StatementAttributeLikeMacro),
                   TT_StatementAttributeLikeMacro});
}

ArrayRef<FormatToken *> FormatTokenLexer::lex() {
  assert(Tokens.empty());
  assert(FirstInLineIndex == 0);
  do {
    Tokens.push_back(getNextToken());
    if (Style.Language == FormatStyle::LK_JavaScript) {
      tryParseJSRegexLiteral();
      handleTemplateStrings();
    }
    if (Style.Language == FormatStyle::LK_TextProto)
      tryParsePythonComment();
    tryMergePreviousTokens();
    if (Style.isCSharp())
      // This needs to come after tokens have been merged so that C#
      // string literals are correctly identified.
      handleCSharpVerbatimAndInterpolatedStrings();
    if (Tokens.back()->NewlinesBefore > 0 || Tokens.back()->IsMultiline)
      FirstInLineIndex = Tokens.size() - 1;
  } while (Tokens.back()->Tok.isNot(tok::eof));
  return Tokens;
}

void FormatTokenLexer::tryMergePreviousTokens() {
  if (tryMerge_TMacro())
    return;
  if (tryMergeConflictMarkers())
    return;
  if (tryMergeLessLess())
    return;
  if (tryMergeForEach())
    return;
  if (Style.isCpp() && tryTransformTryUsageForC())
    return;

  if (Style.Language == FormatStyle::LK_JavaScript || Style.isCSharp()) {
    static const tok::TokenKind NullishCoalescingOperator[] = {tok::question,
                                                               tok::question};
    static const tok::TokenKind NullPropagatingOperator[] = {tok::question,
                                                             tok::period};
    static const tok::TokenKind FatArrow[] = {tok::equal, tok::greater};

    if (tryMergeTokens(FatArrow, TT_FatArrow))
      return;
    if (tryMergeTokens(NullishCoalescingOperator, TT_NullCoalescingOperator)) {
      // Treat like the "||" operator (as opposed to the ternary ?).
      Tokens.back()->Tok.setKind(tok::pipepipe);
      return;
    }
    if (tryMergeTokens(NullPropagatingOperator, TT_NullPropagatingOperator)) {
      // Treat like a regular "." access.
      Tokens.back()->Tok.setKind(tok::period);
      return;
    }
    if (tryMergeNullishCoalescingEqual()) {
      return;
    }
  }

  if (Style.isCSharp()) {
    static const tok::TokenKind CSharpNullConditionalLSquare[] = {
        tok::question, tok::l_square};

    if (tryMergeCSharpKeywordVariables())
      return;
    if (tryMergeCSharpStringLiteral())
      return;
    if (tryTransformCSharpForEach())
      return;
    if (tryMergeTokens(CSharpNullConditionalLSquare,
                       TT_CSharpNullConditionalLSquare)) {
      // Treat like a regular "[" operator.
      Tokens.back()->Tok.setKind(tok::l_square);
      return;
    }
  }

  if (tryMergeNSStringLiteral())
    return;

  if (Style.Language == FormatStyle::LK_JavaScript) {
    static const tok::TokenKind JSIdentity[] = {tok::equalequal, tok::equal};
    static const tok::TokenKind JSNotIdentity[] = {tok::exclaimequal,
                                                   tok::equal};
    static const tok::TokenKind JSShiftEqual[] = {tok::greater, tok::greater,
                                                  tok::greaterequal};
    static const tok::TokenKind JSExponentiation[] = {tok::star, tok::star};
    static const tok::TokenKind JSExponentiationEqual[] = {tok::star,
                                                           tok::starequal};
    static const tok::TokenKind JSPipePipeEqual[] = {tok::pipepipe, tok::equal};
    static const tok::TokenKind JSAndAndEqual[] = {tok::ampamp, tok::equal};

    // FIXME: Investigate what token type gives the correct operator priority.
    if (tryMergeTokens(JSIdentity, TT_BinaryOperator))
      return;
    if (tryMergeTokens(JSNotIdentity, TT_BinaryOperator))
      return;
    if (tryMergeTokens(JSShiftEqual, TT_BinaryOperator))
      return;
    if (tryMergeTokens(JSExponentiation, TT_JsExponentiation))
      return;
    if (tryMergeTokens(JSExponentiationEqual, TT_JsExponentiationEqual)) {
      Tokens.back()->Tok.setKind(tok::starequal);
      return;
    }
    if (tryMergeTokens(JSAndAndEqual, TT_JsAndAndEqual) ||
        tryMergeTokens(JSPipePipeEqual, TT_JsPipePipeEqual)) {
      // Treat like the "=" assignment operator.
      Tokens.back()->Tok.setKind(tok::equal);
      return;
    }
    if (tryMergeJSPrivateIdentifier())
      return;
  }

  if (Style.Language == FormatStyle::LK_Java) {
    static const tok::TokenKind JavaRightLogicalShiftAssign[] = {
        tok::greater, tok::greater, tok::greaterequal};
    if (tryMergeTokens(JavaRightLogicalShiftAssign, TT_BinaryOperator))
      return;
  }
}

bool FormatTokenLexer::tryMergeNSStringLiteral() {
  if (Tokens.size() < 2)
    return false;
  auto &At = *(Tokens.end() - 2);
  auto &String = *(Tokens.end() - 1);
  if (!At->is(tok::at) || !String->is(tok::string_literal))
    return false;
  At->Tok.setKind(tok::string_literal);
  At->TokenText = StringRef(At->TokenText.begin(),
                            String->TokenText.end() - At->TokenText.begin());
  At->ColumnWidth += String->ColumnWidth;
  At->setType(TT_ObjCStringLiteral);
  Tokens.erase(Tokens.end() - 1);
  return true;
}

bool FormatTokenLexer::tryMergeJSPrivateIdentifier() {
  // Merges #idenfier into a single identifier with the text #identifier
  // but the token tok::identifier.
  if (Tokens.size() < 2)
    return false;
  auto &Hash = *(Tokens.end() - 2);
  auto &Identifier = *(Tokens.end() - 1);
  if (!Hash->is(tok::hash) || !Identifier->is(tok::identifier))
    return false;
  Hash->Tok.setKind(tok::identifier);
  Hash->TokenText =
      StringRef(Hash->TokenText.begin(),
                Identifier->TokenText.end() - Hash->TokenText.begin());
  Hash->ColumnWidth += Identifier->ColumnWidth;
  Hash->setType(TT_JsPrivateIdentifier);
  Tokens.erase(Tokens.end() - 1);
  return true;
}

// Search for verbatim or interpolated string literals @"ABC" or
// $"aaaaa{abc}aaaaa" i and mark the token as TT_CSharpStringLiteral, and to
// prevent splitting of @, $ and ".
// Merging of multiline verbatim strings with embedded '"' is handled in
// handleCSharpVerbatimAndInterpolatedStrings with lower-level lexing.
bool FormatTokenLexer::tryMergeCSharpStringLiteral() {
  if (Tokens.size() < 2)
    return false;

  // Interpolated strings could contain { } with " characters inside.
  // $"{x ?? "null"}"
  // should not be split into $"{x ?? ", null, "}" but should treated as a
  // single string-literal.
  //
  // We opt not to try and format expressions inside {} within a C#
  // interpolated string. Formatting expressions within an interpolated string
  // would require similar work as that done for JavaScript template strings
  // in `handleTemplateStrings()`.
  auto &CSharpInterpolatedString = *(Tokens.end() - 2);
  if (CSharpInterpolatedString->getType() == TT_CSharpStringLiteral &&
      (CSharpInterpolatedString->TokenText.startswith(R"($")") ||
       CSharpInterpolatedString->TokenText.startswith(R"($@")"))) {
    int UnmatchedOpeningBraceCount = 0;

    auto TokenTextSize = CSharpInterpolatedString->TokenText.size();
    for (size_t Index = 0; Index < TokenTextSize; ++Index) {
      char C = CSharpInterpolatedString->TokenText[Index];
      if (C == '{') {
        // "{{"  inside an interpolated string is an escaped '{' so skip it.
        if (Index + 1 < TokenTextSize &&
            CSharpInterpolatedString->TokenText[Index + 1] == '{') {
          ++Index;
          continue;
        }
        ++UnmatchedOpeningBraceCount;
      } else if (C == '}') {
        // "}}"  inside an interpolated string is an escaped '}' so skip it.
        if (Index + 1 < TokenTextSize &&
            CSharpInterpolatedString->TokenText[Index + 1] == '}') {
          ++Index;
          continue;
        }
        --UnmatchedOpeningBraceCount;
      }
    }

    if (UnmatchedOpeningBraceCount > 0) {
      auto &NextToken = *(Tokens.end() - 1);
      CSharpInterpolatedString->TokenText =
          StringRef(CSharpInterpolatedString->TokenText.begin(),
                    NextToken->TokenText.end() -
                        CSharpInterpolatedString->TokenText.begin());
      CSharpInterpolatedString->ColumnWidth += NextToken->ColumnWidth;
      Tokens.erase(Tokens.end() - 1);
      return true;
    }
  }

  // Look for @"aaaaaa" or $"aaaaaa".
  auto &String = *(Tokens.end() - 1);
  if (!String->is(tok::string_literal))
    return false;

  auto &At = *(Tokens.end() - 2);
  if (!(At->is(tok::at) || At->TokenText == "$"))
    return false;

  if (Tokens.size() > 2 && At->is(tok::at)) {
    auto &Dollar = *(Tokens.end() - 3);
    if (Dollar->TokenText == "$") {
      // This looks like $@"aaaaa" so we need to combine all 3 tokens.
      Dollar->Tok.setKind(tok::string_literal);
      Dollar->TokenText =
          StringRef(Dollar->TokenText.begin(),
                    String->TokenText.end() - Dollar->TokenText.begin());
      Dollar->ColumnWidth += (At->ColumnWidth + String->ColumnWidth);
      Dollar->setType(TT_CSharpStringLiteral);
      Tokens.erase(Tokens.end() - 2);
      Tokens.erase(Tokens.end() - 1);
      return true;
    }
  }

  // Convert back into just a string_literal.
  At->Tok.setKind(tok::string_literal);
  At->TokenText = StringRef(At->TokenText.begin(),
                            String->TokenText.end() - At->TokenText.begin());
  At->ColumnWidth += String->ColumnWidth;
  At->setType(TT_CSharpStringLiteral);
  Tokens.erase(Tokens.end() - 1);
  return true;
}

// Valid C# attribute targets:
// https://docs.microsoft.com/en-us/dotnet/csharp/programming-guide/concepts/attributes/#attribute-targets
const llvm::StringSet<> FormatTokenLexer::CSharpAttributeTargets = {
    "assembly", "module",   "field",  "event", "method",
    "param",    "property", "return", "type",
};

bool FormatTokenLexer::tryMergeNullishCoalescingEqual() {
  if (Tokens.size() < 2)
    return false;
  auto &NullishCoalescing = *(Tokens.end() - 2);
  auto &Equal = *(Tokens.end() - 1);
  if (NullishCoalescing->getType() != TT_NullCoalescingOperator ||
      !Equal->is(tok::equal))
    return false;
  NullishCoalescing->Tok.setKind(tok::equal); // no '??=' in clang tokens.
  NullishCoalescing->TokenText =
      StringRef(NullishCoalescing->TokenText.begin(),
                Equal->TokenText.end() - NullishCoalescing->TokenText.begin());
  NullishCoalescing->ColumnWidth += Equal->ColumnWidth;
  NullishCoalescing->setType(TT_NullCoalescingEqual);
  Tokens.erase(Tokens.end() - 1);
  return true;
}

bool FormatTokenLexer::tryMergeCSharpKeywordVariables() {
  if (Tokens.size() < 2)
    return false;
  auto &At = *(Tokens.end() - 2);
  auto &Keyword = *(Tokens.end() - 1);
  if (!At->is(tok::at))
    return false;
  if (!Keywords.isCSharpKeyword(*Keyword))
    return false;

  At->Tok.setKind(tok::identifier);
  At->TokenText = StringRef(At->TokenText.begin(),
                            Keyword->TokenText.end() - At->TokenText.begin());
  At->ColumnWidth += Keyword->ColumnWidth;
  At->setType(Keyword->getType());
  Tokens.erase(Tokens.end() - 1);
  return true;
}

// In C# transform identifier foreach into kw_foreach
bool FormatTokenLexer::tryTransformCSharpForEach() {
  if (Tokens.size() < 1)
    return false;
  auto &Identifier = *(Tokens.end() - 1);
  if (!Identifier->is(tok::identifier))
    return false;
  if (Identifier->TokenText != "foreach")
    return false;

  Identifier->setType(TT_ForEachMacro);
  Identifier->Tok.setKind(tok::kw_for);
  return true;
}

bool FormatTokenLexer::tryMergeForEach() {
  if (Tokens.size() < 2)
    return false;
  auto &For = *(Tokens.end() - 2);
  auto &Each = *(Tokens.end() - 1);
  if (!For->is(tok::kw_for))
    return false;
  if (!Each->is(tok::identifier))
    return false;
  if (Each->TokenText != "each")
    return false;

  For->setType(TT_ForEachMacro);
  For->Tok.setKind(tok::kw_for);

  For->TokenText = StringRef(For->TokenText.begin(),
                             Each->TokenText.end() - For->TokenText.begin());
  For->ColumnWidth += Each->ColumnWidth;
  Tokens.erase(Tokens.end() - 1);
  return true;
}

bool FormatTokenLexer::tryTransformTryUsageForC() {
  if (Tokens.size() < 2)
    return false;
  auto &Try = *(Tokens.end() - 2);
  if (!Try->is(tok::kw_try))
    return false;
  auto &Next = *(Tokens.end() - 1);
  if (Next->isOneOf(tok::l_brace, tok::colon, tok::hash, tok::comment))
    return false;

  if (Tokens.size() > 2) {
    auto &At = *(Tokens.end() - 3);
    if (At->is(tok::at))
      return false;
  }

  Try->Tok.setKind(tok::identifier);
  return true;
}

bool FormatTokenLexer::tryMergeLessLess() {
  // Merge X,less,less,Y into X,lessless,Y unless X or Y is less.
  if (Tokens.size() < 3)
    return false;

  bool FourthTokenIsLess = false;
  if (Tokens.size() > 3)
    FourthTokenIsLess = (Tokens.end() - 4)[0]->is(tok::less);

  auto First = Tokens.end() - 3;
  if (First[2]->is(tok::less) || First[1]->isNot(tok::less) ||
      First[0]->isNot(tok::less) || FourthTokenIsLess)
    return false;

  // Only merge if there currently is no whitespace between the two "<".
  if (First[1]->WhitespaceRange.getBegin() !=
      First[1]->WhitespaceRange.getEnd())
    return false;

  First[0]->Tok.setKind(tok::lessless);
  First[0]->TokenText = "<<";
  First[0]->ColumnWidth += 1;
  Tokens.erase(Tokens.end() - 2);
  return true;
}

bool FormatTokenLexer::tryMergeTokens(ArrayRef<tok::TokenKind> Kinds,
                                      TokenType NewType) {
  if (Tokens.size() < Kinds.size())
    return false;

  SmallVectorImpl<FormatToken *>::const_iterator First =
      Tokens.end() - Kinds.size();
  if (!First[0]->is(Kinds[0]))
    return false;
  unsigned AddLength = 0;
  for (unsigned i = 1; i < Kinds.size(); ++i) {
    if (!First[i]->is(Kinds[i]) || First[i]->WhitespaceRange.getBegin() !=
                                       First[i]->WhitespaceRange.getEnd())
      return false;
    AddLength += First[i]->TokenText.size();
  }
  Tokens.resize(Tokens.size() - Kinds.size() + 1);
  First[0]->TokenText = StringRef(First[0]->TokenText.data(),
                                  First[0]->TokenText.size() + AddLength);
  First[0]->ColumnWidth += AddLength;
  First[0]->setType(NewType);
  return true;
}

// Returns \c true if \p Tok can only be followed by an operand in JavaScript.
bool FormatTokenLexer::precedesOperand(FormatToken *Tok) {
  // NB: This is not entirely correct, as an r_paren can introduce an operand
  // location in e.g. `if (foo) /bar/.exec(...);`. That is a rare enough
  // corner case to not matter in practice, though.
  return Tok->isOneOf(tok::period, tok::l_paren, tok::comma, tok::l_brace,
                      tok::r_brace, tok::l_square, tok::semi, tok::exclaim,
                      tok::colon, tok::question, tok::tilde) ||
         Tok->isOneOf(tok::kw_return, tok::kw_do, tok::kw_case, tok::kw_throw,
                      tok::kw_else, tok::kw_new, tok::kw_delete, tok::kw_void,
                      tok::kw_typeof, Keywords.kw_instanceof, Keywords.kw_in) ||
         Tok->isBinaryOperator();
}

bool FormatTokenLexer::canPrecedeRegexLiteral(FormatToken *Prev) {
  if (!Prev)
    return true;

  // Regex literals can only follow after prefix unary operators, not after
  // postfix unary operators. If the '++' is followed by a non-operand
  // introducing token, the slash here is the operand and not the start of a
  // regex.
  // `!` is an unary prefix operator, but also a post-fix operator that casts
  // away nullability, so the same check applies.
  if (Prev->isOneOf(tok::plusplus, tok::minusminus, tok::exclaim))
    return (Tokens.size() < 3 || precedesOperand(Tokens[Tokens.size() - 3]));

  // The previous token must introduce an operand location where regex
  // literals can occur.
  if (!precedesOperand(Prev))
    return false;

  return true;
}

// Tries to parse a JavaScript Regex literal starting at the current token,
// if that begins with a slash and is in a location where JavaScript allows
// regex literals. Changes the current token to a regex literal and updates
// its text if successful.
void FormatTokenLexer::tryParseJSRegexLiteral() {
  FormatToken *RegexToken = Tokens.back();
  if (!RegexToken->isOneOf(tok::slash, tok::slashequal))
    return;

  FormatToken *Prev = nullptr;
  for (auto I = Tokens.rbegin() + 1, E = Tokens.rend(); I != E; ++I) {
    // NB: Because previous pointers are not initialized yet, this cannot use
    // Token.getPreviousNonComment.
    if ((*I)->isNot(tok::comment)) {
      Prev = *I;
      break;
    }
  }

  if (!canPrecedeRegexLiteral(Prev))
    return;

  // 'Manually' lex ahead in the current file buffer.
  const char *Offset = Lex->getBufferLocation();
  const char *RegexBegin = Offset - RegexToken->TokenText.size();
  StringRef Buffer = Lex->getBuffer();
  bool InCharacterClass = false;
  bool HaveClosingSlash = false;
  for (; !HaveClosingSlash && Offset != Buffer.end(); ++Offset) {
    // Regular expressions are terminated with a '/', which can only be
    // escaped using '\' or a character class between '[' and ']'.
    // See http://www.ecma-international.org/ecma-262/5.1/#sec-7.8.5.
    switch (*Offset) {
    case '\\':
      // Skip the escaped character.
      ++Offset;
      break;
    case '[':
      InCharacterClass = true;
      break;
    case ']':
      InCharacterClass = false;
      break;
    case '/':
      if (!InCharacterClass)
        HaveClosingSlash = true;
      break;
    }
  }

  RegexToken->setType(TT_RegexLiteral);
  // Treat regex literals like other string_literals.
  RegexToken->Tok.setKind(tok::string_literal);
  RegexToken->TokenText = StringRef(RegexBegin, Offset - RegexBegin);
  RegexToken->ColumnWidth = RegexToken->TokenText.size();

  resetLexer(SourceMgr.getFileOffset(Lex->getSourceLocation(Offset)));
}

void FormatTokenLexer::handleCSharpVerbatimAndInterpolatedStrings() {
  FormatToken *CSharpStringLiteral = Tokens.back();

  if (CSharpStringLiteral->getType() != TT_CSharpStringLiteral)
    return;

  // Deal with multiline strings.
  if (!(CSharpStringLiteral->TokenText.startswith(R"(@")") ||
        CSharpStringLiteral->TokenText.startswith(R"($@")")))
    return;

  const char *StrBegin =
      Lex->getBufferLocation() - CSharpStringLiteral->TokenText.size();
  const char *Offset = StrBegin;
  if (CSharpStringLiteral->TokenText.startswith(R"(@")"))
    Offset += 2;
  else // CSharpStringLiteral->TokenText.startswith(R"($@")")
    Offset += 3;

  // Look for a terminating '"' in the current file buffer.
  // Make no effort to format code within an interpolated or verbatim string.
  for (; Offset != Lex->getBuffer().end(); ++Offset) {
    if (Offset[0] == '"') {
      // "" within a verbatim string is an escaped double quote: skip it.
      if (Offset + 1 < Lex->getBuffer().end() && Offset[1] == '"')
        ++Offset;
      else
        break;
    }
  }

  // Make no attempt to format code properly if a verbatim string is
  // unterminated.
  if (Offset == Lex->getBuffer().end())
    return;

  StringRef LiteralText(StrBegin, Offset - StrBegin + 1);
  CSharpStringLiteral->TokenText = LiteralText;

  // Adjust width for potentially multiline string literals.
  size_t FirstBreak = LiteralText.find('\n');
  StringRef FirstLineText = FirstBreak == StringRef::npos
                                ? LiteralText
                                : LiteralText.substr(0, FirstBreak);
  CSharpStringLiteral->ColumnWidth = encoding::columnWidthWithTabs(
      FirstLineText, CSharpStringLiteral->OriginalColumn, Style.TabWidth,
      Encoding);
  size_t LastBreak = LiteralText.rfind('\n');
  if (LastBreak != StringRef::npos) {
    CSharpStringLiteral->IsMultiline = true;
    unsigned StartColumn = 0;
    CSharpStringLiteral->LastLineColumnWidth = encoding::columnWidthWithTabs(
        LiteralText.substr(LastBreak + 1, LiteralText.size()), StartColumn,
        Style.TabWidth, Encoding);
  }

  SourceLocation loc = Offset < Lex->getBuffer().end()
                           ? Lex->getSourceLocation(Offset + 1)
                           : SourceMgr.getLocForEndOfFile(ID);
  resetLexer(SourceMgr.getFileOffset(loc));
}

void FormatTokenLexer::handleTemplateStrings() {
  FormatToken *BacktickToken = Tokens.back();

  if (BacktickToken->is(tok::l_brace)) {
    StateStack.push(LexerState::NORMAL);
    return;
  }
  if (BacktickToken->is(tok::r_brace)) {
    if (StateStack.size() == 1)
      return;
    StateStack.pop();
    if (StateStack.top() != LexerState::TEMPLATE_STRING)
      return;
    // If back in TEMPLATE_STRING, fallthrough and continue parsing the
  } else if (BacktickToken->is(tok::unknown) &&
             BacktickToken->TokenText == "`") {
    StateStack.push(LexerState::TEMPLATE_STRING);
  } else {
    return; // Not actually a template
  }

  // 'Manually' lex ahead in the current file buffer.
  const char *Offset = Lex->getBufferLocation();
  const char *TmplBegin = Offset - BacktickToken->TokenText.size(); // at "`"
  for (; Offset != Lex->getBuffer().end(); ++Offset) {
    if (Offset[0] == '`') {
      StateStack.pop();
      break;
    }
    if (Offset[0] == '\\') {
      ++Offset; // Skip the escaped character.
    } else if (Offset + 1 < Lex->getBuffer().end() && Offset[0] == '$' &&
               Offset[1] == '{') {
      // '${' introduces an expression interpolation in the template string.
      StateStack.push(LexerState::NORMAL);
      ++Offset;
      break;
    }
  }

  StringRef LiteralText(TmplBegin, Offset - TmplBegin + 1);
  BacktickToken->setType(TT_TemplateString);
  BacktickToken->Tok.setKind(tok::string_literal);
  BacktickToken->TokenText = LiteralText;

  // Adjust width for potentially multiline string literals.
  size_t FirstBreak = LiteralText.find('\n');
  StringRef FirstLineText = FirstBreak == StringRef::npos
                                ? LiteralText
                                : LiteralText.substr(0, FirstBreak);
  BacktickToken->ColumnWidth = encoding::columnWidthWithTabs(
      FirstLineText, BacktickToken->OriginalColumn, Style.TabWidth, Encoding);
  size_t LastBreak = LiteralText.rfind('\n');
  if (LastBreak != StringRef::npos) {
    BacktickToken->IsMultiline = true;
    unsigned StartColumn = 0; // The template tail spans the entire line.
    BacktickToken->LastLineColumnWidth = encoding::columnWidthWithTabs(
        LiteralText.substr(LastBreak + 1, LiteralText.size()), StartColumn,
        Style.TabWidth, Encoding);
  }

  SourceLocation loc = Offset < Lex->getBuffer().end()
                           ? Lex->getSourceLocation(Offset + 1)
                           : SourceMgr.getLocForEndOfFile(ID);
  resetLexer(SourceMgr.getFileOffset(loc));
}

void FormatTokenLexer::tryParsePythonComment() {
  FormatToken *HashToken = Tokens.back();
  if (!HashToken->isOneOf(tok::hash, tok::hashhash))
    return;
  // Turn the remainder of this line into a comment.
  const char *CommentBegin =
      Lex->getBufferLocation() - HashToken->TokenText.size(); // at "#"
  size_t From = CommentBegin - Lex->getBuffer().begin();
  size_t To = Lex->getBuffer().find_first_of('\n', From);
  if (To == StringRef::npos)
    To = Lex->getBuffer().size();
  size_t Len = To - From;
  HashToken->setType(TT_LineComment);
  HashToken->Tok.setKind(tok::comment);
  HashToken->TokenText = Lex->getBuffer().substr(From, Len);
  SourceLocation Loc = To < Lex->getBuffer().size()
                           ? Lex->getSourceLocation(CommentBegin + Len)
                           : SourceMgr.getLocForEndOfFile(ID);
  resetLexer(SourceMgr.getFileOffset(Loc));
}

bool FormatTokenLexer::tryMerge_TMacro() {
  if (Tokens.size() < 4)
    return false;
  FormatToken *Last = Tokens.back();
  if (!Last->is(tok::r_paren))
    return false;

  FormatToken *String = Tokens[Tokens.size() - 2];
  if (!String->is(tok::string_literal) || String->IsMultiline)
    return false;

  if (!Tokens[Tokens.size() - 3]->is(tok::l_paren))
    return false;

  FormatToken *Macro = Tokens[Tokens.size() - 4];
  if (Macro->TokenText != "_T")
    return false;

  const char *Start = Macro->TokenText.data();
  const char *End = Last->TokenText.data() + Last->TokenText.size();
  String->TokenText = StringRef(Start, End - Start);
  String->IsFirst = Macro->IsFirst;
  String->LastNewlineOffset = Macro->LastNewlineOffset;
  String->WhitespaceRange = Macro->WhitespaceRange;
  String->OriginalColumn = Macro->OriginalColumn;
  String->ColumnWidth = encoding::columnWidthWithTabs(
      String->TokenText, String->OriginalColumn, Style.TabWidth, Encoding);
  String->NewlinesBefore = Macro->NewlinesBefore;
  String->HasUnescapedNewline = Macro->HasUnescapedNewline;

  Tokens.pop_back();
  Tokens.pop_back();
  Tokens.pop_back();
  Tokens.back() = String;
  return true;
}

bool FormatTokenLexer::tryMergeConflictMarkers() {
  if (Tokens.back()->NewlinesBefore == 0 && Tokens.back()->isNot(tok::eof))
    return false;

  // Conflict lines look like:
  // <marker> <text from the vcs>
  // For example:
  // >>>>>>> /file/in/file/system at revision 1234
  //
  // We merge all tokens in a line that starts with a conflict marker
  // into a single token with a special token type that the unwrapped line
  // parser will use to correctly rebuild the underlying code.

  FileID ID;
  // Get the position of the first token in the line.
  unsigned FirstInLineOffset;
  std::tie(ID, FirstInLineOffset) = SourceMgr.getDecomposedLoc(
      Tokens[FirstInLineIndex]->getStartOfNonWhitespace());
  StringRef Buffer = SourceMgr.getBufferOrFake(ID).getBuffer();
  // Calculate the offset of the start of the current line.
  auto LineOffset = Buffer.rfind('\n', FirstInLineOffset);
  if (LineOffset == StringRef::npos) {
    LineOffset = 0;
  } else {
    ++LineOffset;
  }

  auto FirstSpace = Buffer.find_first_of(" \n", LineOffset);
  StringRef LineStart;
  if (FirstSpace == StringRef::npos) {
    LineStart = Buffer.substr(LineOffset);
  } else {
    LineStart = Buffer.substr(LineOffset, FirstSpace - LineOffset);
  }

  TokenType Type = TT_Unknown;
  if (LineStart == "<<<<<<<" || LineStart == ">>>>") {
    Type = TT_ConflictStart;
  } else if (LineStart == "|||||||" || LineStart == "=======" ||
             LineStart == "====") {
    Type = TT_ConflictAlternative;
  } else if (LineStart == ">>>>>>>" || LineStart == "<<<<") {
    Type = TT_ConflictEnd;
  }

  if (Type != TT_Unknown) {
    FormatToken *Next = Tokens.back();

    Tokens.resize(FirstInLineIndex + 1);
    // We do not need to build a complete token here, as we will skip it
    // during parsing anyway (as we must not touch whitespace around conflict
    // markers).
    Tokens.back()->setType(Type);
    Tokens.back()->Tok.setKind(tok::kw___unknown_anytype);

    Tokens.push_back(Next);
    return true;
  }

  return false;
}

FormatToken *FormatTokenLexer::getStashedToken() {
  // Create a synthesized second '>' or '<' token.
  Token Tok = FormatTok->Tok;
  StringRef TokenText = FormatTok->TokenText;

  unsigned OriginalColumn = FormatTok->OriginalColumn;
  FormatTok = new (Allocator.Allocate()) FormatToken;
  FormatTok->Tok = Tok;
  SourceLocation TokLocation =
      FormatTok->Tok.getLocation().getLocWithOffset(Tok.getLength() - 1);
  FormatTok->Tok.setLocation(TokLocation);
  FormatTok->WhitespaceRange = SourceRange(TokLocation, TokLocation);
  FormatTok->TokenText = TokenText;
  FormatTok->ColumnWidth = 1;
  FormatTok->OriginalColumn = OriginalColumn + 1;

  return FormatTok;
}

FormatToken *FormatTokenLexer::getNextToken() {
  if (StateStack.top() == LexerState::TOKEN_STASHED) {
    StateStack.pop();
    return getStashedToken();
  }

  FormatTok = new (Allocator.Allocate()) FormatToken;
  readRawToken(*FormatTok);
  SourceLocation WhitespaceStart =
      FormatTok->Tok.getLocation().getLocWithOffset(-TrailingWhitespace);
  FormatTok->IsFirst = IsFirstToken;
  IsFirstToken = false;

  // Consume and record whitespace until we find a significant token.
  unsigned WhitespaceLength = TrailingWhitespace;
  while (FormatTok->Tok.is(tok::unknown)) {
    StringRef Text = FormatTok->TokenText;
    auto EscapesNewline = [&](int pos) {
      // A '\r' here is just part of '\r\n'. Skip it.
      if (pos >= 0 && Text[pos] == '\r')
        --pos;
      // See whether there is an odd number of '\' before this.
      // FIXME: This is wrong. A '\' followed by a newline is always removed,
      // regardless of whether there is another '\' before it.
      // FIXME: Newlines can also be escaped by a '?' '?' '/' trigraph.
      unsigned count = 0;
      for (; pos >= 0; --pos, ++count)
        if (Text[pos] != '\\')
          break;
      return count & 1;
    };
    // FIXME: This miscounts tok:unknown tokens that are not just
    // whitespace, e.g. a '`' character.
    for (int i = 0, e = Text.size(); i != e; ++i) {
      switch (Text[i]) {
      case '\n':
        ++FormatTok->NewlinesBefore;
        FormatTok->HasUnescapedNewline = !EscapesNewline(i - 1);
        FormatTok->LastNewlineOffset = WhitespaceLength + i + 1;
        Column = 0;
        break;
      case '\r':
        FormatTok->LastNewlineOffset = WhitespaceLength + i + 1;
        Column = 0;
        break;
      case '\f':
      case '\v':
        Column = 0;
        break;
      case ' ':
        ++Column;
        break;
      case '\t':
        Column +=
            Style.TabWidth - (Style.TabWidth ? Column % Style.TabWidth : 0);
        break;
      case '\\':
        if (i + 1 == e || (Text[i + 1] != '\r' && Text[i + 1] != '\n'))
          FormatTok->setType(TT_ImplicitStringLiteral);
        break;
      default:
        FormatTok->setType(TT_ImplicitStringLiteral);
        break;
      }
      if (FormatTok->getType() == TT_ImplicitStringLiteral)
        break;
    }

    if (FormatTok->is(TT_ImplicitStringLiteral))
      break;
    WhitespaceLength += FormatTok->Tok.getLength();

    readRawToken(*FormatTok);
  }

  // JavaScript and Java do not allow to escape the end of the line with a
  // backslash. Backslashes are syntax errors in plain source, but can occur in
  // comments. When a single line comment ends with a \, it'll cause the next
  // line of code to be lexed as a comment, breaking formatting. The code below
  // finds comments that contain a backslash followed by a line break, truncates
  // the comment token at the backslash, and resets the lexer to restart behind
  // the backslash.
  if ((Style.Language == FormatStyle::LK_JavaScript ||
       Style.Language == FormatStyle::LK_Java) &&
      FormatTok->is(tok::comment) && FormatTok->TokenText.startswith("//")) {
    size_t BackslashPos = FormatTok->TokenText.find('\\');
    while (BackslashPos != StringRef::npos) {
      if (BackslashPos + 1 < FormatTok->TokenText.size() &&
          FormatTok->TokenText[BackslashPos + 1] == '\n') {
        const char *Offset = Lex->getBufferLocation();
        Offset -= FormatTok->TokenText.size();
        Offset += BackslashPos + 1;
        resetLexer(SourceMgr.getFileOffset(Lex->getSourceLocation(Offset)));
        FormatTok->TokenText = FormatTok->TokenText.substr(0, BackslashPos + 1);
        FormatTok->ColumnWidth = encoding::columnWidthWithTabs(
            FormatTok->TokenText, FormatTok->OriginalColumn, Style.TabWidth,
            Encoding);
        break;
      }
      BackslashPos = FormatTok->TokenText.find('\\', BackslashPos + 1);
    }
  }

  // In case the token starts with escaped newlines, we want to
  // take them into account as whitespace - this pattern is quite frequent
  // in macro definitions.
  // FIXME: Add a more explicit test.
  while (FormatTok->TokenText.size() > 1 && FormatTok->TokenText[0] == '\\') {
    unsigned SkippedWhitespace = 0;
    if (FormatTok->TokenText.size() > 2 &&
        (FormatTok->TokenText[1] == '\r' && FormatTok->TokenText[2] == '\n'))
      SkippedWhitespace = 3;
    else if (FormatTok->TokenText[1] == '\n')
      SkippedWhitespace = 2;
    else
      break;

    ++FormatTok->NewlinesBefore;
    WhitespaceLength += SkippedWhitespace;
    FormatTok->LastNewlineOffset = SkippedWhitespace;
    Column = 0;
    FormatTok->TokenText = FormatTok->TokenText.substr(SkippedWhitespace);
  }

  FormatTok->WhitespaceRange = SourceRange(
      WhitespaceStart, WhitespaceStart.getLocWithOffset(WhitespaceLength));

  FormatTok->OriginalColumn = Column;

  TrailingWhitespace = 0;
  if (FormatTok->Tok.is(tok::comment)) {
    // FIXME: Add the trimmed whitespace to Column.
    StringRef UntrimmedText = FormatTok->TokenText;
    FormatTok->TokenText = FormatTok->TokenText.rtrim(" \t\v\f");
    TrailingWhitespace = UntrimmedText.size() - FormatTok->TokenText.size();
  } else if (FormatTok->Tok.is(tok::raw_identifier)) {
    IdentifierInfo &Info = IdentTable.get(FormatTok->TokenText);
    FormatTok->Tok.setIdentifierInfo(&Info);
    FormatTok->Tok.setKind(Info.getTokenID());
    if (Style.Language == FormatStyle::LK_Java &&
        FormatTok->isOneOf(tok::kw_struct, tok::kw_union, tok::kw_delete,
                           tok::kw_operator)) {
      FormatTok->Tok.setKind(tok::identifier);
      FormatTok->Tok.setIdentifierInfo(nullptr);
    } else if (Style.Language == FormatStyle::LK_JavaScript &&
               FormatTok->isOneOf(tok::kw_struct, tok::kw_union,
                                  tok::kw_operator)) {
      FormatTok->Tok.setKind(tok::identifier);
      FormatTok->Tok.setIdentifierInfo(nullptr);
    }
  } else if (FormatTok->Tok.is(tok::greatergreater)) {
    FormatTok->Tok.setKind(tok::greater);
    FormatTok->TokenText = FormatTok->TokenText.substr(0, 1);
    ++Column;
    StateStack.push(LexerState::TOKEN_STASHED);
  } else if (FormatTok->Tok.is(tok::lessless)) {
    FormatTok->Tok.setKind(tok::less);
    FormatTok->TokenText = FormatTok->TokenText.substr(0, 1);
    ++Column;
    StateStack.push(LexerState::TOKEN_STASHED);
  }

  // Now FormatTok is the next non-whitespace token.

  StringRef Text = FormatTok->TokenText;
  size_t FirstNewlinePos = Text.find('\n');
  if (FirstNewlinePos == StringRef::npos) {
    // FIXME: ColumnWidth actually depends on the start column, we need to
    // take this into account when the token is moved.
    FormatTok->ColumnWidth =
        encoding::columnWidthWithTabs(Text, Column, Style.TabWidth, Encoding);
    Column += FormatTok->ColumnWidth;
  } else {
    FormatTok->IsMultiline = true;
    // FIXME: ColumnWidth actually depends on the start column, we need to
    // take this into account when the token is moved.
    FormatTok->ColumnWidth = encoding::columnWidthWithTabs(
        Text.substr(0, FirstNewlinePos), Column, Style.TabWidth, Encoding);

    // The last line of the token always starts in column 0.
    // Thus, the length can be precomputed even in the presence of tabs.
    FormatTok->LastLineColumnWidth = encoding::columnWidthWithTabs(
        Text.substr(Text.find_last_of('\n') + 1), 0, Style.TabWidth, Encoding);
    Column = FormatTok->LastLineColumnWidth;
  }

  if (Style.isCpp()) {
    auto it = Macros.find(FormatTok->Tok.getIdentifierInfo());
    if (!(Tokens.size() > 0 && Tokens.back()->Tok.getIdentifierInfo() &&
          Tokens.back()->Tok.getIdentifierInfo()->getPPKeywordID() ==
              tok::pp_define) &&
        it != Macros.end()) {
      FormatTok->setType(it->second);
      if (it->second == TT_IfMacro) {
        // The lexer token currently has type tok::kw_unknown. However, for this
        // substitution to be treated correctly in the TokenAnnotator, faking
        // the tok value seems to be needed. Not sure if there's a more elegant
        // way.
        FormatTok->Tok.setKind(tok::kw_if);
      }
    } else if (FormatTok->is(tok::identifier)) {
      if (MacroBlockBeginRegex.match(Text)) {
        FormatTok->setType(TT_MacroBlockBegin);
      } else if (MacroBlockEndRegex.match(Text)) {
        FormatTok->setType(TT_MacroBlockEnd);
      }
    }
  }

  return FormatTok;
}

void FormatTokenLexer::readRawToken(FormatToken &Tok) {
  Lex->LexFromRawLexer(Tok.Tok);
  Tok.TokenText = StringRef(SourceMgr.getCharacterData(Tok.Tok.getLocation()),
                            Tok.Tok.getLength());
  // For formatting, treat unterminated string literals like normal string
  // literals.
  if (Tok.is(tok::unknown)) {
    if (!Tok.TokenText.empty() && Tok.TokenText[0] == '"') {
      Tok.Tok.setKind(tok::string_literal);
      Tok.IsUnterminatedLiteral = true;
    } else if (Style.Language == FormatStyle::LK_JavaScript &&
               Tok.TokenText == "''") {
      Tok.Tok.setKind(tok::string_literal);
    }
  }

  if ((Style.Language == FormatStyle::LK_JavaScript ||
       Style.Language == FormatStyle::LK_Proto ||
       Style.Language == FormatStyle::LK_TextProto) &&
      Tok.is(tok::char_constant)) {
    Tok.Tok.setKind(tok::string_literal);
  }

  if (Tok.is(tok::comment) && (Tok.TokenText == "// clang-format on" ||
                               Tok.TokenText == "/* clang-format on */")) {
    FormattingDisabled = false;
  }

  Tok.Finalized = FormattingDisabled;

  if (Tok.is(tok::comment) && (Tok.TokenText == "// clang-format off" ||
                               Tok.TokenText == "/* clang-format off */")) {
    FormattingDisabled = true;
  }
}

void FormatTokenLexer::resetLexer(unsigned Offset) {
  StringRef Buffer = SourceMgr.getBufferData(ID);
  Lex.reset(new Lexer(SourceMgr.getLocForStartOfFile(ID),
                      getFormattingLangOpts(Style), Buffer.begin(),
                      Buffer.begin() + Offset, Buffer.end()));
  Lex->SetKeepWhitespaceMode(true);
  TrailingWhitespace = 0;
}

} // namespace format
} // namespace clang
