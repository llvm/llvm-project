//===-- lib/Parser/prescan.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "prescan.h"
#include "preprocessor.h"
#include "token-sequence.h"
#include "flang/Common/idioms.h"
#include "flang/Parser/characters.h"
#include "flang/Parser/message.h"
#include "flang/Parser/source.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <cstring>
#include <utility>
#include <vector>

namespace Fortran::parser {

using common::LanguageFeature;

static constexpr int maxPrescannerNesting{100};

Prescanner::Prescanner(Messages &messages, CookedSource &cooked,
    Preprocessor &preprocessor, common::LanguageFeatureControl lfc)
    : messages_{messages}, cooked_{cooked}, preprocessor_{preprocessor},
      allSources_{preprocessor_.allSources()}, features_{lfc},
      encoding_{allSources_.encoding()} {}

Prescanner::Prescanner(const Prescanner &that)
    : messages_{that.messages_}, cooked_{that.cooked_},
      preprocessor_{that.preprocessor_}, allSources_{that.allSources_},
      features_{that.features_}, inFixedForm_{that.inFixedForm_},
      fixedFormColumnLimit_{that.fixedFormColumnLimit_},
      encoding_{that.encoding_}, prescannerNesting_{that.prescannerNesting_ +
                                     1},
      skipLeadingAmpersand_{that.skipLeadingAmpersand_},
      compilerDirectiveBloomFilter_{that.compilerDirectiveBloomFilter_},
      compilerDirectiveSentinels_{that.compilerDirectiveSentinels_} {}

static inline constexpr bool IsFixedFormCommentChar(char ch) {
  return ch == '!' || ch == '*' || ch == 'C' || ch == 'c';
}

static void NormalizeCompilerDirectiveCommentMarker(TokenSequence &dir) {
  char *p{dir.GetMutableCharData()};
  char *limit{p + dir.SizeInChars()};
  for (; p < limit; ++p) {
    if (*p != ' ') {
      CHECK(IsFixedFormCommentChar(*p));
      *p = '!';
      return;
    }
  }
  DIE("compiler directive all blank");
}

void Prescanner::Prescan(ProvenanceRange range) {
  startProvenance_ = range.start();
  start_ = allSources_.GetSource(range);
  CHECK(start_);
  limit_ = start_ + range.size();
  nextLine_ = start_;
  const bool beganInFixedForm{inFixedForm_};
  if (prescannerNesting_ > maxPrescannerNesting) {
    Say(GetProvenance(start_),
        "too many nested INCLUDE/#include files, possibly circular"_err_en_US);
    return;
  }
  while (!IsAtEnd()) {
    Statement();
  }
  if (inFixedForm_ != beganInFixedForm) {
    std::string dir{"!dir$ "};
    if (beganInFixedForm) {
      dir += "fixed";
    } else {
      dir += "free";
    }
    dir += '\n';
    TokenSequence tokens{dir, allSources_.AddCompilerInsertion(dir).start()};
    tokens.Emit(cooked_);
  }
}

void Prescanner::Statement() {
  TokenSequence tokens;
  const char *statementStart{nextLine_};
  LineClassification line{ClassifyLine(statementStart)};
  switch (line.kind) {
  case LineClassification::Kind::Comment:
    nextLine_ += line.payloadOffset; // advance to '!' or newline
    NextLine();
    return;
  case LineClassification::Kind::IncludeLine:
    FortranInclude(nextLine_ + line.payloadOffset);
    NextLine();
    return;
  case LineClassification::Kind::ConditionalCompilationDirective:
  case LineClassification::Kind::IncludeDirective:
  case LineClassification::Kind::DefinitionDirective:
  case LineClassification::Kind::PreprocessorDirective:
    preprocessor_.Directive(TokenizePreprocessorDirective(), *this);
    return;
  case LineClassification::Kind::CompilerDirective: {
    directiveSentinel_ = line.sentinel;
    CHECK(InCompilerDirective());
    BeginStatementAndAdvance();
    if (inFixedForm_) {
      CHECK(IsFixedFormCommentChar(*at_));
    } else {
      while (*at_ == ' ' || *at_ == '\t') {
        ++at_, ++column_;
      }
      CHECK(*at_ == '!');
    }
    std::optional<int> condOffset;
    if (directiveSentinel_[0] == '$' && directiveSentinel_[1] == '\0') {
      // OpenMP conditional compilation line.
      condOffset = 2;
    } else if (directiveSentinel_[0] == '@' && directiveSentinel_[1] == 'c' &&
        directiveSentinel_[2] == 'u' && directiveSentinel_[3] == 'f' &&
        directiveSentinel_[4] == '\0') {
      // CUDA conditional compilation line.
      condOffset = 5;
    }
    if (condOffset) {
      at_ += *condOffset, column_ += *condOffset;
      if (auto payload{IsIncludeLine(at_)}) {
        FortranInclude(at_ + *payload);
        return;
      } else if (inFixedForm_) {
        LabelField(tokens);
      } else {
        SkipSpaces();
      }
    } else {
      // Compiler directive.  Emit normalized sentinel, squash following spaces.
      EmitChar(tokens, '!');
      ++at_, ++column_;
      for (const char *sp{directiveSentinel_}; *sp != '\0';
           ++sp, ++at_, ++column_) {
        EmitChar(tokens, *sp);
      }
      if (*at_ == ' ' || *at_ == '\t') {
        EmitChar(tokens, ' ');
        while (*at_ == ' ' || *at_ == '\t') {
          ++at_, ++column_;
        }
      }
      tokens.CloseToken();
    }
    break;
  }
  case LineClassification::Kind::Source:
    BeginStatementAndAdvance();
    if (inFixedForm_) {
      if (features_.IsEnabled(LanguageFeature::OldDebugLines) &&
          (*at_ == 'D' || *at_ == 'd')) {
        NextChar();
      }
      LabelField(tokens);
    } else if (skipLeadingAmpersand_) {
      skipLeadingAmpersand_ = false;
      const char *p{SkipWhiteSpace(at_)};
      if (p < limit_ && *p == '&') {
        column_ += ++p - at_;
        at_ = p;
      }
    } else {
      SkipSpaces();
    }
    break;
  }

  while (NextToken(tokens)) {
  }
  if (continuationLines_ > 255) {
    if (features_.ShouldWarn(common::LanguageFeature::MiscSourceExtensions)) {
      Say(GetProvenance(statementStart),
          "%d continuation lines is more than the Fortran standard allows"_port_en_US,
          continuationLines_);
    }
  }

  Provenance newlineProvenance{GetCurrentProvenance()};
  if (std::optional<TokenSequence> preprocessed{
          preprocessor_.MacroReplacement(tokens, *this)}) {
    // Reprocess the preprocessed line.  Append a newline temporarily.
    preprocessed->PutNextTokenChar('\n', newlineProvenance);
    preprocessed->CloseToken();
    const char *ppd{preprocessed->ToCharBlock().begin()};
    LineClassification ppl{ClassifyLine(ppd)};
    preprocessed->pop_back(); // remove the newline
    switch (ppl.kind) {
    case LineClassification::Kind::Comment:
      break;
    case LineClassification::Kind::IncludeLine:
      FortranInclude(ppd + ppl.payloadOffset);
      break;
    case LineClassification::Kind::ConditionalCompilationDirective:
    case LineClassification::Kind::IncludeDirective:
    case LineClassification::Kind::DefinitionDirective:
    case LineClassification::Kind::PreprocessorDirective:
      Say(preprocessed->GetProvenanceRange(),
          "Preprocessed line resembles a preprocessor directive"_warn_en_US);
      preprocessed->ToLowerCase()
          .CheckBadFortranCharacters(messages_, *this)
          .CheckBadParentheses(messages_)
          .Emit(cooked_);
      break;
    case LineClassification::Kind::CompilerDirective:
      if (preprocessed->HasRedundantBlanks()) {
        preprocessed->RemoveRedundantBlanks();
      }
      while (CompilerDirectiveContinuation(*preprocessed, ppl.sentinel)) {
        newlineProvenance = GetCurrentProvenance();
      }
      NormalizeCompilerDirectiveCommentMarker(*preprocessed);
      preprocessed->ToLowerCase();
      SourceFormChange(preprocessed->ToString());
      preprocessed->ClipComment(*this, true /* skip first ! */)
          .CheckBadFortranCharacters(messages_, *this)
          .CheckBadParentheses(messages_)
          .Emit(cooked_);
      break;
    case LineClassification::Kind::Source:
      if (inFixedForm_) {
        if (preprocessed->HasBlanks(/*after column*/ 6)) {
          preprocessed->RemoveBlanks(/*after column*/ 6);
        }
      } else {
        while (SourceLineContinuation(*preprocessed)) {
          newlineProvenance = GetCurrentProvenance();
        }
        if (preprocessed->HasRedundantBlanks()) {
          preprocessed->RemoveRedundantBlanks();
        }
      }
      preprocessed->ToLowerCase()
          .ClipComment(*this)
          .CheckBadFortranCharacters(messages_, *this)
          .CheckBadParentheses(messages_)
          .Emit(cooked_);
      break;
    }
  } else {
    if (line.kind == LineClassification::Kind::CompilerDirective) {
      while (CompilerDirectiveContinuation(tokens, line.sentinel)) {
        newlineProvenance = GetCurrentProvenance();
      }
      tokens.ToLowerCase();
      SourceFormChange(tokens.ToString());
    } else { // Kind::Source
      tokens.ToLowerCase();
      if (inFixedForm_) {
        EnforceStupidEndStatementRules(tokens);
      }
    }
    tokens.CheckBadFortranCharacters(messages_, *this)
        .CheckBadParentheses(messages_)
        .Emit(cooked_);
  }
  if (omitNewline_) {
    omitNewline_ = false;
  } else {
    cooked_.Put('\n', newlineProvenance);
  }
  directiveSentinel_ = nullptr;
}

TokenSequence Prescanner::TokenizePreprocessorDirective() {
  CHECK(!IsAtEnd() && !inPreprocessorDirective_);
  inPreprocessorDirective_ = true;
  BeginStatementAndAdvance();
  TokenSequence tokens;
  while (NextToken(tokens)) {
  }
  inPreprocessorDirective_ = false;
  return tokens;
}

void Prescanner::NextLine() {
  void *vstart{static_cast<void *>(const_cast<char *>(nextLine_))};
  void *v{std::memchr(vstart, '\n', limit_ - nextLine_)};
  if (!v) {
    nextLine_ = limit_;
  } else {
    const char *nl{const_cast<const char *>(static_cast<char *>(v))};
    nextLine_ = nl + 1;
  }
}

void Prescanner::LabelField(TokenSequence &token) {
  int outCol{1};
  const char *start{at_};
  std::optional<int> badColumn;
  for (; *at_ != '\n' && column_ <= 6; ++at_) {
    if (*at_ == '\t') {
      ++at_;
      column_ = 7;
      break;
    }
    if (*at_ != ' ' &&
        !(*at_ == '0' && column_ == 6)) { // '0' in column 6 becomes space
      EmitChar(token, *at_);
      ++outCol;
      if (!badColumn && (column_ == 6 || !IsDecimalDigit(*at_))) {
        badColumn = column_;
      }
    }
    ++column_;
  }
  if (badColumn && !preprocessor_.IsNameDefined(token.CurrentOpenToken())) {
    Say(GetProvenance(start + *badColumn - 1),
        *badColumn == 6
            ? "Statement should not begin with a continuation line"_warn_en_US
            : "Character in fixed-form label field must be a digit"_warn_en_US);
    token.clear();
    if (*badColumn < 6) {
      at_ = start;
      column_ = 1;
      return;
    }
    outCol = 1;
  }
  if (outCol == 1) { // empty label field
    // Emit a space so that, if the line is rescanned after preprocessing,
    // a leading 'C' or 'D' won't be left-justified and then accidentally
    // misinterpreted as a comment card.
    EmitChar(token, ' ');
    ++outCol;
  }
  token.CloseToken();
  SkipToNextSignificantCharacter();
  if (IsDecimalDigit(*at_)) {
    if (features_.ShouldWarn(common::LanguageFeature::MiscSourceExtensions)) {
      Say(GetCurrentProvenance(),
          "Label digit is not in fixed-form label field"_port_en_US);
    }
  }
}

// 6.3.3.5: A program unit END statement, or any other statement whose
// initial line resembles an END statement, shall not be continued in
// fixed form source.
void Prescanner::EnforceStupidEndStatementRules(const TokenSequence &tokens) {
  CharBlock cBlock{tokens.ToCharBlock()};
  const char *str{cBlock.begin()};
  std::size_t n{cBlock.size()};
  if (n < 3) {
    return;
  }
  std::size_t j{0};
  for (; j < n && (str[j] == ' ' || (str[j] >= '0' && str[j] <= '9')); ++j) {
  }
  if (j + 3 > n || std::memcmp(str + j, "end", 3) != 0) {
    return;
  }
  // It starts with END, possibly after a label.
  auto start{allSources_.GetSourcePosition(tokens.GetCharProvenance(j))};
  auto end{allSources_.GetSourcePosition(tokens.GetCharProvenance(n - 1))};
  if (!start || !end) {
    return;
  }
  if (&*start->sourceFile == &*end->sourceFile && start->line == end->line) {
    return; // no continuation
  }
  j += 3;
  static const char *const prefixes[]{"program", "subroutine", "function",
      "blockdata", "module", "submodule", nullptr};
  bool isPrefix{j == n || !IsLegalInIdentifier(str[j])}; // prefix is END
  std::size_t endOfPrefix{j - 1};
  for (const char *const *p{prefixes}; *p; ++p) {
    std::size_t pLen{std::strlen(*p)};
    if (j + pLen <= n && std::memcmp(str + j, *p, pLen) == 0) {
      isPrefix = true; // END thing as prefix
      j += pLen;
      endOfPrefix = j - 1;
      for (; j < n && IsLegalInIdentifier(str[j]); ++j) {
      }
      break;
    }
  }
  if (isPrefix) {
    auto range{tokens.GetTokenProvenanceRange(1)};
    if (j == n) { // END or END thing [name]
      Say(range,
          "Program unit END statement may not be continued in fixed form source"_err_en_US);
    } else {
      auto endOfPrefixPos{
          allSources_.GetSourcePosition(tokens.GetCharProvenance(endOfPrefix))};
      auto next{allSources_.GetSourcePosition(tokens.GetCharProvenance(j))};
      if (endOfPrefixPos && next &&
          &*endOfPrefixPos->sourceFile == &*start->sourceFile &&
          endOfPrefixPos->line == start->line &&
          (&*next->sourceFile != &*start->sourceFile ||
              next->line != start->line)) {
        Say(range,
            "Initial line of continued statement must not appear to be a program unit END in fixed form source"_err_en_US);
      }
    }
  }
}

void Prescanner::SkipToEndOfLine() {
  while (*at_ != '\n') {
    ++at_, ++column_;
  }
}

bool Prescanner::MustSkipToEndOfLine() const {
  if (inFixedForm_ && column_ > fixedFormColumnLimit_ && !tabInCurrentLine_) {
    return true; // skip over ignored columns in right margin (73:80)
  } else if (*at_ == '!' && !inCharLiteral_) {
    return true; // inline comment goes to end of source line
  } else {
    return false;
  }
}

void Prescanner::NextChar() {
  CHECK(*at_ != '\n');
  ++at_, ++column_;
  while (at_[0] == '\xef' && at_[1] == '\xbb' && at_[2] == '\xbf') {
    // UTF-8 byte order mark - treat this file as UTF-8
    at_ += 3;
    encoding_ = Encoding::UTF_8;
  }
  SkipToNextSignificantCharacter();
}

// Skip everything that should be ignored until the next significant
// character is reached; handles C-style comments in preprocessing
// directives, Fortran ! comments, stuff after the right margin in
// fixed form, and all forms of line continuation.
void Prescanner::SkipToNextSignificantCharacter() {
  if (inPreprocessorDirective_) {
    SkipCComments();
  } else {
    bool mightNeedSpace{false};
    if (MustSkipToEndOfLine()) {
      SkipToEndOfLine();
    } else {
      mightNeedSpace = *at_ == '\n';
    }
    for (; Continuation(mightNeedSpace); mightNeedSpace = false) {
      ++continuationLines_;
      if (MustSkipToEndOfLine()) {
        SkipToEndOfLine();
      }
    }
    if (*at_ == '\t') {
      tabInCurrentLine_ = true;
    }
  }
}

void Prescanner::SkipCComments() {
  while (true) {
    if (IsCComment(at_)) {
      if (const char *after{SkipCComment(at_)}) {
        column_ += after - at_;
        // May have skipped over one or more newlines; relocate the start of
        // the next line.
        nextLine_ = at_ = after;
        NextLine();
      } else {
        // Don't emit any messages about unclosed C-style comments, because
        // the sequence /* can appear legally in a FORMAT statement.  There's
        // no ambiguity, since the sequence */ cannot appear legally.
        break;
      }
    } else if (inPreprocessorDirective_ && at_[0] == '\\' && at_ + 2 < limit_ &&
        at_[1] == '\n' && !IsAtEnd()) {
      BeginSourceLineAndAdvance();
    } else {
      break;
    }
  }
}

void Prescanner::SkipSpaces() {
  while (*at_ == ' ' || *at_ == '\t') {
    NextChar();
  }
  insertASpace_ = false;
}

const char *Prescanner::SkipWhiteSpace(const char *p) {
  while (*p == ' ' || *p == '\t') {
    ++p;
  }
  return p;
}

const char *Prescanner::SkipWhiteSpaceAndCComments(const char *p) const {
  while (true) {
    if (*p == ' ' || *p == '\t') {
      ++p;
    } else if (IsCComment(p)) {
      if (const char *after{SkipCComment(p)}) {
        p = after;
      } else {
        break;
      }
    } else {
      break;
    }
  }
  return p;
}

const char *Prescanner::SkipCComment(const char *p) const {
  char star{' '}, slash{' '};
  p += 2;
  while (star != '*' || slash != '/') {
    if (p >= limit_) {
      return nullptr; // signifies an unterminated comment
    }
    star = slash;
    slash = *p++;
  }
  return p;
}

bool Prescanner::NextToken(TokenSequence &tokens) {
  CHECK(at_ >= start_ && at_ < limit_);
  if (InFixedFormSource()) {
    SkipSpaces();
  } else {
    if (*at_ == '/' && IsCComment(at_)) {
      // Recognize and skip over classic C style /*comments*/ when
      // outside a character literal.
      if (features_.ShouldWarn(LanguageFeature::ClassicCComments)) {
        Say(GetCurrentProvenance(),
            "nonstandard usage: C-style comment"_port_en_US);
      }
      SkipCComments();
    }
    if (*at_ == ' ' || *at_ == '\t') {
      // Compress free-form white space into a single space character.
      const auto theSpace{at_};
      char previous{at_ <= start_ ? ' ' : at_[-1]};
      NextChar();
      SkipSpaces();
      if (*at_ == '\n') {
        // Discard white space at the end of a line.
      } else if (!inPreprocessorDirective_ &&
          (previous == '(' || *at_ == '(' || *at_ == ')')) {
        // Discard white space before/after '(' and before ')', unless in a
        // preprocessor directive.  This helps yield space-free contiguous
        // names for generic interfaces like OPERATOR( + ) and
        // READ ( UNFORMATTED ), without misinterpreting #define f (notAnArg).
        // This has the effect of silently ignoring the illegal spaces in
        // the array constructor ( /1,2/ ) but that seems benign; it's
        // hard to avoid that while still removing spaces from OPERATOR( / )
        // and OPERATOR( // ).
      } else {
        // Preserve the squashed white space as a single space character.
        tokens.PutNextTokenChar(' ', GetProvenance(theSpace));
        tokens.CloseToken();
        return true;
      }
    }
  }
  if (insertASpace_) {
    tokens.PutNextTokenChar(' ', spaceProvenance_);
    insertASpace_ = false;
  }
  if (*at_ == '\n') {
    return false;
  }
  const char *start{at_};
  if (*at_ == '\'' || *at_ == '"') {
    QuotedCharacterLiteral(tokens, start);
    preventHollerith_ = false;
  } else if (IsDecimalDigit(*at_)) {
    int n{0}, digits{0};
    static constexpr int maxHollerith{256 /*lines*/ * (132 - 6 /*columns*/)};
    do {
      if (n < maxHollerith) {
        n = 10 * n + DecimalDigitValue(*at_);
      }
      EmitCharAndAdvance(tokens, *at_);
      ++digits;
      if (InFixedFormSource()) {
        SkipSpaces();
      }
    } while (IsDecimalDigit(*at_));
    if ((*at_ == 'h' || *at_ == 'H') && n > 0 && n < maxHollerith &&
        !preventHollerith_) {
      Hollerith(tokens, n, start);
    } else if (*at_ == '.') {
      while (IsDecimalDigit(EmitCharAndAdvance(tokens, *at_))) {
      }
      ExponentAndKind(tokens);
    } else if (ExponentAndKind(tokens)) {
    } else if (digits == 1 && n == 0 && (*at_ == 'x' || *at_ == 'X') &&
        inPreprocessorDirective_) {
      do {
        EmitCharAndAdvance(tokens, *at_);
      } while (IsHexadecimalDigit(*at_));
    } else if (IsLetter(*at_)) {
      // Handles FORMAT(3I9HHOLLERITH) by skipping over the first I so that
      // we don't misrecognize I9HOLLERITH as an identifier in the next case.
      EmitCharAndAdvance(tokens, *at_);
    } else if (at_[0] == '_' && (at_[1] == '\'' || at_[1] == '"')) { // 4_"..."
      EmitCharAndAdvance(tokens, *at_);
      QuotedCharacterLiteral(tokens, start);
    }
    preventHollerith_ = false;
  } else if (*at_ == '.') {
    char nch{EmitCharAndAdvance(tokens, '.')};
    if (!inPreprocessorDirective_ && IsDecimalDigit(nch)) {
      while (IsDecimalDigit(EmitCharAndAdvance(tokens, *at_))) {
      }
      ExponentAndKind(tokens);
    } else if (nch == '.' && EmitCharAndAdvance(tokens, '.') == '.') {
      EmitCharAndAdvance(tokens, '.'); // variadic macro definition ellipsis
    }
    preventHollerith_ = false;
  } else if (IsLegalInIdentifier(*at_)) {
    while (IsLegalInIdentifier(EmitCharAndAdvance(tokens, *at_))) {
    }
    if (InFixedFormSource()) {
      SkipSpaces();
    }
    if ((*at_ == '\'' || *at_ == '"') &&
        tokens.CharAt(tokens.SizeInChars() - 1) == '_') { // kind_"..."
      QuotedCharacterLiteral(tokens, start);
      preventHollerith_ = false;
    } else {
      preventHollerith_ = true; // DO 10 H = ...
    }
  } else if (*at_ == '*') {
    if (EmitCharAndAdvance(tokens, '*') == '*') {
      EmitCharAndAdvance(tokens, '*');
    } else {
      // Subtle ambiguity:
      //  CHARACTER*2H     declares H because *2 is a kind specifier
      //  DATAC/N*2H  /    is repeated Hollerith
      preventHollerith_ = !slashInCurrentStatement_;
    }
  } else {
    char ch{*at_};
    if (ch == '(') {
      if (parenthesisNesting_++ == 0) {
        isPossibleMacroCall_ = tokens.SizeInTokens() > 0 &&
            preprocessor_.IsFunctionLikeDefinition(
                tokens.TokenAt(tokens.SizeInTokens() - 1));
      }
    } else if (ch == ')' && parenthesisNesting_ > 0) {
      --parenthesisNesting_;
    }
    char nch{EmitCharAndAdvance(tokens, ch)};
    preventHollerith_ = false;
    if ((nch == '=' &&
            (ch == '<' || ch == '>' || ch == '/' || ch == '=' || ch == '!')) ||
        (ch == nch &&
            (ch == '/' || ch == ':' || ch == '*' || ch == '#' || ch == '&' ||
                ch == '|' || ch == '<' || ch == '>')) ||
        (ch == '=' && nch == '>')) {
      // token comprises two characters
      EmitCharAndAdvance(tokens, nch);
    } else if (ch == '/') {
      slashInCurrentStatement_ = true;
    } else if (ch == ';' && InFixedFormSource()) {
      SkipSpaces();
      if (IsDecimalDigit(*at_)) {
        if (features_.ShouldWarn(
                common::LanguageFeature::MiscSourceExtensions)) {
          Say(GetProvenanceRange(at_, at_ + 1),
              "Label should be in the label field"_port_en_US);
        }
      }
    }
  }
  tokens.CloseToken();
  return true;
}

bool Prescanner::ExponentAndKind(TokenSequence &tokens) {
  char ed{ToLowerCaseLetter(*at_)};
  if (ed != 'e' && ed != 'd') {
    return false;
  }
  EmitCharAndAdvance(tokens, ed);
  if (*at_ == '+' || *at_ == '-') {
    EmitCharAndAdvance(tokens, *at_);
  }
  while (IsDecimalDigit(*at_)) {
    EmitCharAndAdvance(tokens, *at_);
  }
  if (*at_ == '_') {
    while (IsLegalInIdentifier(EmitCharAndAdvance(tokens, *at_))) {
    }
  }
  return true;
}

void Prescanner::QuotedCharacterLiteral(
    TokenSequence &tokens, const char *start) {
  char quote{*at_};
  const char *end{at_ + 1};
  inCharLiteral_ = true;
  continuationInCharLiteral_ = true;
  const auto emit{[&](char ch) { EmitChar(tokens, ch); }};
  const auto insert{[&](char ch) { EmitInsertedChar(tokens, ch); }};
  bool isEscaped{false};
  bool escapesEnabled{features_.IsEnabled(LanguageFeature::BackslashEscapes)};
  while (true) {
    if (*at_ == '\\') {
      if (escapesEnabled) {
        isEscaped = !isEscaped;
      } else {
        // The parser always processes escape sequences, so don't confuse it
        // when escapes are disabled.
        insert('\\');
      }
    } else {
      isEscaped = false;
    }
    EmitQuotedChar(static_cast<unsigned char>(*at_), emit, insert, false,
        Encoding::LATIN_1);
    while (PadOutCharacterLiteral(tokens)) {
    }
    if (*at_ == '\n') {
      if (!inPreprocessorDirective_) {
        Say(GetProvenanceRange(start, end),
            "Incomplete character literal"_err_en_US);
      }
      break;
    }
    end = at_ + 1;
    NextChar();
    if (*at_ == quote && !isEscaped) {
      // A doubled unescaped quote mark becomes a single instance of that
      // quote character in the literal (later).  There can be spaces between
      // the quotes in fixed form source.
      EmitChar(tokens, quote);
      inCharLiteral_ = false; // for cases like print *, '...'!comment
      NextChar();
      if (InFixedFormSource()) {
        SkipSpaces();
      }
      if (*at_ != quote) {
        break;
      }
      inCharLiteral_ = true;
    }
  }
  continuationInCharLiteral_ = false;
  inCharLiteral_ = false;
}

void Prescanner::Hollerith(
    TokenSequence &tokens, int count, const char *start) {
  inCharLiteral_ = true;
  CHECK(*at_ == 'h' || *at_ == 'H');
  EmitChar(tokens, 'H');
  while (count-- > 0) {
    if (PadOutCharacterLiteral(tokens)) {
    } else if (*at_ == '\n') {
      Say(GetProvenanceRange(start, at_),
          "Possible truncated Hollerith literal"_warn_en_US);
      break;
    } else {
      NextChar();
      // Each multi-byte character encoding counts as a single character.
      // No escape sequences are recognized.
      // Hollerith is always emitted to the cooked character
      // stream in UTF-8.
      DecodedCharacter decoded{DecodeCharacter(
          encoding_, at_, static_cast<std::size_t>(limit_ - at_), false)};
      if (decoded.bytes > 0) {
        EncodedCharacter utf8{
            EncodeCharacter<Encoding::UTF_8>(decoded.codepoint)};
        for (int j{0}; j < utf8.bytes; ++j) {
          EmitChar(tokens, utf8.buffer[j]);
        }
        at_ += decoded.bytes - 1;
      } else {
        Say(GetProvenanceRange(start, at_),
            "Bad character in Hollerith literal"_err_en_US);
        break;
      }
    }
  }
  if (*at_ != '\n') {
    NextChar();
  }
  inCharLiteral_ = false;
}

// In fixed form, source card images must be processed as if they were at
// least 72 columns wide, at least in character literal contexts.
bool Prescanner::PadOutCharacterLiteral(TokenSequence &tokens) {
  while (inFixedForm_ && !tabInCurrentLine_ && at_[1] == '\n') {
    if (column_ < fixedFormColumnLimit_) {
      tokens.PutNextTokenChar(' ', spaceProvenance_);
      ++column_;
      return true;
    }
    if (!FixedFormContinuation(false /*no need to insert space*/) ||
        tabInCurrentLine_) {
      return false;
    }
    CHECK(column_ == 7);
    --at_; // point to column 6 of continuation line
    column_ = 6;
  }
  return false;
}

static bool IsAtProcess(const char *p) {
  static const char pAtProc[]{"process"};
  for (std::size_t i{0}; i < sizeof pAtProc - 1; ++i) {
    if (ToLowerCaseLetter(*++p) != pAtProc[i])
      return false;
  }
  return true;
}

bool Prescanner::IsFixedFormCommentLine(const char *start) const {
  const char *p{start};

  // The @process directive must start in column 1.
  if (*p == '@' && IsAtProcess(p)) {
    return true;
  }

  if (IsFixedFormCommentChar(*p) || *p == '%' || // VAX %list, %eject, &c.
      ((*p == 'D' || *p == 'd') &&
          !features_.IsEnabled(LanguageFeature::OldDebugLines))) {
    return true;
  }
  bool anyTabs{false};
  while (true) {
    if (*p == ' ') {
      ++p;
    } else if (*p == '\t') {
      anyTabs = true;
      ++p;
    } else if (*p == '0' && !anyTabs && p == start + 5) {
      ++p; // 0 in column 6 must treated as a space
    } else {
      break;
    }
  }
  if (!anyTabs && p >= start + fixedFormColumnLimit_) {
    return true;
  }
  if (*p == '!' && !inCharLiteral_ && (anyTabs || p != start + 5)) {
    return true;
  }
  return *p == '\n';
}

const char *Prescanner::IsFreeFormComment(const char *p) const {
  p = SkipWhiteSpaceAndCComments(p);
  if (*p == '!' || *p == '\n') {
    return p;
  } else if (*p == '@') {
    return IsAtProcess(p) ? p : nullptr;
  } else {
    return nullptr;
  }
}

std::optional<std::size_t> Prescanner::IsIncludeLine(const char *start) const {
  const char *p{SkipWhiteSpace(start)};
  if (*p == '0' && inFixedForm_ && p == start + 5) {
    // Accept "     0INCLUDE" in fixed form.
    p = SkipWhiteSpace(p + 1);
  }
  for (const char *q{"include"}; *q; ++q) {
    if (ToLowerCaseLetter(*p) != *q) {
      return std::nullopt;
    }
    p = SkipWhiteSpace(p + 1);
  }
  if (IsDecimalDigit(*p)) { // accept & ignore a numeric kind prefix
    for (p = SkipWhiteSpace(p + 1); IsDecimalDigit(*p);
         p = SkipWhiteSpace(p + 1)) {
    }
    if (*p != '_') {
      return std::nullopt;
    }
    p = SkipWhiteSpace(p + 1);
  }
  if (*p == '"' || *p == '\'') {
    return {p - start};
  }
  return std::nullopt;
}

void Prescanner::FortranInclude(const char *firstQuote) {
  const char *p{firstQuote};
  while (*p != '"' && *p != '\'') {
    ++p;
  }
  char quote{*p};
  std::string path;
  for (++p; *p != '\n'; ++p) {
    if (*p == quote) {
      if (p[1] != quote) {
        break;
      }
      ++p;
    }
    path += *p;
  }
  if (*p != quote) {
    Say(GetProvenanceRange(firstQuote, p),
        "malformed path name string"_err_en_US);
    return;
  }
  p = SkipWhiteSpace(p + 1);
  if (*p != '\n' && *p != '!') {
    const char *garbage{p};
    for (; *p != '\n' && *p != '!'; ++p) {
    }
    Say(GetProvenanceRange(garbage, p),
        "excess characters after path name"_warn_en_US);
  }
  std::string buf;
  llvm::raw_string_ostream error{buf};
  Provenance provenance{GetProvenance(nextLine_)};
  std::optional<std::string> prependPath;
  if (const SourceFile * currentFile{allSources_.GetSourceFile(provenance)}) {
    prependPath = DirectoryName(currentFile->path());
  }
  const SourceFile *included{
      allSources_.Open(path, error, std::move(prependPath))};
  if (!included) {
    Say(provenance, "INCLUDE: %s"_err_en_US, error.str());
  } else if (included->bytes() > 0) {
    ProvenanceRange includeLineRange{
        provenance, static_cast<std::size_t>(p - nextLine_)};
    ProvenanceRange fileRange{
        allSources_.AddIncludedFile(*included, includeLineRange)};
    Prescanner{*this}.set_encoding(included->encoding()).Prescan(fileRange);
  }
}

const char *Prescanner::IsPreprocessorDirectiveLine(const char *start) const {
  const char *p{start};
  for (; *p == ' '; ++p) {
  }
  if (*p == '#') {
    if (inFixedForm_ && p == start + 5) {
      return nullptr;
    }
  } else {
    p = SkipWhiteSpace(p);
    if (*p != '#') {
      return nullptr;
    }
  }
  return SkipWhiteSpace(p + 1);
}

bool Prescanner::IsNextLinePreprocessorDirective() const {
  return IsPreprocessorDirectiveLine(nextLine_) != nullptr;
}

bool Prescanner::SkipCommentLine(bool afterAmpersand) {
  if (IsAtEnd()) {
    if (afterAmpersand && prescannerNesting_ > 0) {
      // A continuation marker at the end of the last line in an
      // include file inhibits the newline for that line.
      SkipToEndOfLine();
      omitNewline_ = true;
    }
    return false;
  }
  auto lineClass{ClassifyLine(nextLine_)};
  if (lineClass.kind == LineClassification::Kind::Comment) {
    NextLine();
    return true;
  } else if (inPreprocessorDirective_) {
    return false;
  } else if (lineClass.kind ==
          LineClassification::Kind::ConditionalCompilationDirective ||
      lineClass.kind == LineClassification::Kind::PreprocessorDirective) {
    // Allow conditional compilation directives (e.g., #ifdef) to affect
    // continuation lines.
    // Allow other preprocessor directives, too, except #include
    // (when it does not follow '&'), #define, and #undef (because
    // they cannot be allowed to affect preceding text on a
    // continued line).
    preprocessor_.Directive(TokenizePreprocessorDirective(), *this);
    return true;
  } else if (afterAmpersand &&
      (lineClass.kind == LineClassification::Kind::IncludeDirective ||
          lineClass.kind == LineClassification::Kind::IncludeLine)) {
    SkipToEndOfLine();
    omitNewline_ = true;
    skipLeadingAmpersand_ = true;
    return false;
  } else {
    return false;
  }
}

const char *Prescanner::FixedFormContinuationLine(bool mightNeedSpace) {
  if (IsAtEnd()) {
    return nullptr;
  }
  tabInCurrentLine_ = false;
  char col1{*nextLine_};
  if (IsFixedFormCommentChar(col1)) {
    int j{1};
    if (InCompilerDirective()) {
      // Must be a continued compiler directive.
      for (; j < 5; ++j) {
        char ch{directiveSentinel_[j - 1]};
        if (ch == '\0') {
          break;
        }
        if (ch != ToLowerCaseLetter(nextLine_[j])) {
          return nullptr;
        }
      }
    } else if (features_.IsEnabled(LanguageFeature::OpenMP)) {
      // Fixed Source Form Conditional Compilation Sentinels.
      if (nextLine_[1] != '$') {
        return nullptr;
      }
      j++;
    } else {
      return nullptr;
    }
    for (; j < 5; ++j) {
      if (nextLine_[j] != ' ') {
        return nullptr;
      }
    }
    char col6{nextLine_[5]};
    if (col6 != '\n' && col6 != '\t' && col6 != ' ' && col6 != '0') {
      if (nextLine_[6] != ' ' && mightNeedSpace) {
        insertASpace_ = true;
      }
      return nextLine_ + 6;
    }
    return nullptr;
  } else {
    // Normal case: not in a compiler directive.
    if (col1 == '&' &&
        features_.IsEnabled(
            LanguageFeature::FixedFormContinuationWithColumn1Ampersand)) {
      // Extension: '&' as continuation marker
      if (features_.ShouldWarn(
              LanguageFeature::FixedFormContinuationWithColumn1Ampersand)) {
        Say(GetProvenance(nextLine_), "nonstandard usage"_port_en_US);
      }
      return nextLine_ + 1;
    }
    if (col1 == '\t' && nextLine_[1] >= '1' && nextLine_[1] <= '9') {
      tabInCurrentLine_ = true;
      return nextLine_ + 2; // VAX extension
    }
    if ((col1 == ' ' ||
            ((col1 == 'D' || col1 == 'd') &&
                features_.IsEnabled(LanguageFeature::OldDebugLines))) &&
        nextLine_[1] == ' ' && nextLine_[2] == ' ' && nextLine_[3] == ' ' &&
        nextLine_[4] == ' ') {
      char col6{nextLine_[5]};
      if (col6 != '\n' && col6 != '\t' && col6 != ' ' && col6 != '0') {
        if ((col6 == 'i' || col6 == 'I') && IsIncludeLine(nextLine_)) {
          // It's An INCLUDE line, not a continuation
        } else {
          return nextLine_ + 6;
        }
      }
    }
    if (IsImplicitContinuation()) {
      return nextLine_;
    }
  }
  return nullptr; // not a continuation line
}

const char *Prescanner::FreeFormContinuationLine(bool ampersand) {
  const char *p{nextLine_};
  if (p >= limit_) {
    return nullptr;
  }
  p = SkipWhiteSpace(p);
  if (InCompilerDirective()) {
    if (*p++ != '!') {
      return nullptr;
    }
    for (const char *s{directiveSentinel_}; *s != '\0'; ++p, ++s) {
      if (*s != ToLowerCaseLetter(*p)) {
        return nullptr;
      }
    }
    p = SkipWhiteSpace(p);
    if (*p == '&') {
      if (!ampersand) {
        insertASpace_ = true;
      }
      return p + 1;
    } else if (ampersand) {
      return p;
    } else {
      return nullptr;
    }
  } else {
    if (*p == '&') {
      return p + 1;
    } else if (*p == '!' || *p == '\n' || *p == '#') {
      return nullptr;
    } else if (ampersand || IsImplicitContinuation()) {
      if (continuationInCharLiteral_) {
        // 'a'&            -> 'a''b' == "a'b"
        //   'b'
        if (features_.ShouldWarn(
                common::LanguageFeature::MiscSourceExtensions)) {
          Say(GetProvenanceRange(p, p + 1),
              "Character literal continuation line should have been preceded by '&'"_port_en_US);
        }
      } else if (p > nextLine_) {
        --p;
      } else {
        insertASpace_ = true;
      }
      return p;
    } else {
      return nullptr;
    }
  }
}

bool Prescanner::FixedFormContinuation(bool mightNeedSpace) {
  // N.B. We accept '&' as a continuation indicator in fixed form, too,
  // but not in a character literal.
  if (*at_ == '&' && inCharLiteral_) {
    return false;
  }
  do {
    if (const char *cont{FixedFormContinuationLine(mightNeedSpace)}) {
      BeginSourceLine(cont);
      column_ = 7;
      NextLine();
      return true;
    }
  } while (SkipCommentLine(false /* not after ampersand */));
  return false;
}

bool Prescanner::FreeFormContinuation() {
  const char *p{at_};
  bool ampersand{*p == '&'};
  if (ampersand) {
    p = SkipWhiteSpace(p + 1);
  }
  if (*p != '\n') {
    if (inCharLiteral_) {
      return false;
    } else if (*p == '!') { // & ! comment - ok
    } else if (ampersand && isPossibleMacroCall_ && (*p == ',' || *p == ')')) {
      return false; // allow & at end of a macro argument
    } else if (features_.ShouldWarn(LanguageFeature::CruftAfterAmpersand)) {
      Say(GetProvenance(p), "missing ! before comment after &"_warn_en_US);
    }
  }
  do {
    if (const char *cont{FreeFormContinuationLine(ampersand)}) {
      BeginSourceLine(cont);
      NextLine();
      return true;
    }
  } while (SkipCommentLine(ampersand));
  return false;
}

// Implicit line continuation allows a preprocessor macro call with
// arguments to span multiple lines.
bool Prescanner::IsImplicitContinuation() const {
  return !inPreprocessorDirective_ && !inCharLiteral_ && isPossibleMacroCall_ &&
      parenthesisNesting_ > 0 && !IsAtEnd() &&
      ClassifyLine(nextLine_).kind == LineClassification::Kind::Source;
}

bool Prescanner::Continuation(bool mightNeedFixedFormSpace) {
  if (*at_ == '\n' || *at_ == '&') {
    if (inFixedForm_) {
      return FixedFormContinuation(mightNeedFixedFormSpace);
    } else {
      return FreeFormContinuation();
    }
  } else {
    return false;
  }
}

std::optional<Prescanner::LineClassification>
Prescanner::IsFixedFormCompilerDirectiveLine(const char *start) const {
  const char *p{start};
  char col1{*p++};
  if (!IsFixedFormCommentChar(col1)) {
    return std::nullopt;
  }
  char sentinel[5], *sp{sentinel};
  int column{2};
  for (; column < 6; ++column, ++p) {
    if (*p == ' ' || *p == '\n' || *p == '\t') {
      break;
    }
    if (sp == sentinel + 1 && sentinel[0] == '$' && IsDecimalDigit(*p)) {
      // OpenMP conditional compilation line: leave the label alone
      break;
    }
    *sp++ = ToLowerCaseLetter(*p);
  }
  if (column == 6) {
    if (*p == ' ' || *p == '\t' || *p == '0') {
      ++p;
    } else {
      // This is a Continuation line, not an initial directive line.
      return std::nullopt;
    }
  }
  if (sp == sentinel) {
    return std::nullopt;
  }
  *sp = '\0';
  if (const char *ss{IsCompilerDirectiveSentinel(
          sentinel, static_cast<std::size_t>(sp - sentinel))}) {
    std::size_t payloadOffset = p - start;
    return {LineClassification{
        LineClassification::Kind::CompilerDirective, payloadOffset, ss}};
  }
  return std::nullopt;
}

std::optional<Prescanner::LineClassification>
Prescanner::IsFreeFormCompilerDirectiveLine(const char *start) const {
  char sentinel[8];
  const char *p{SkipWhiteSpace(start)};
  if (*p++ != '!') {
    return std::nullopt;
  }
  for (std::size_t j{0}; j + 1 < sizeof sentinel; ++p, ++j) {
    if (*p == '\n') {
      break;
    }
    if (*p == ' ' || *p == '\t' || *p == '&') {
      if (j == 0) {
        break;
      }
      sentinel[j] = '\0';
      p = SkipWhiteSpace(p + 1);
      if (*p == '!') {
        break;
      }
      if (const char *sp{IsCompilerDirectiveSentinel(sentinel, j)}) {
        std::size_t offset = p - start;
        return {LineClassification{
            LineClassification::Kind::CompilerDirective, offset, sp}};
      }
      break;
    }
    sentinel[j] = ToLowerCaseLetter(*p);
  }
  return std::nullopt;
}

Prescanner &Prescanner::AddCompilerDirectiveSentinel(const std::string &dir) {
  std::uint64_t packed{0};
  for (char ch : dir) {
    packed = (packed << 8) | (ToLowerCaseLetter(ch) & 0xff);
  }
  compilerDirectiveBloomFilter_.set(packed % prime1);
  compilerDirectiveBloomFilter_.set(packed % prime2);
  compilerDirectiveSentinels_.insert(dir);
  return *this;
}

const char *Prescanner::IsCompilerDirectiveSentinel(
    const char *sentinel, std::size_t len) const {
  std::uint64_t packed{0};
  for (std::size_t j{0}; j < len; ++j) {
    packed = (packed << 8) | (sentinel[j] & 0xff);
  }
  if (len == 0 || !compilerDirectiveBloomFilter_.test(packed % prime1) ||
      !compilerDirectiveBloomFilter_.test(packed % prime2)) {
    return nullptr;
  }
  const auto iter{compilerDirectiveSentinels_.find(std::string(sentinel, len))};
  return iter == compilerDirectiveSentinels_.end() ? nullptr : iter->c_str();
}

const char *Prescanner::IsCompilerDirectiveSentinel(CharBlock token) const {
  const char *p{token.begin()};
  const char *end{p + token.size()};
  while (p < end && (*p == ' ' || *p == '\n')) {
    ++p;
  }
  if (p < end && *p == '!') {
    ++p;
  }
  while (end > p && (end[-1] == ' ' || end[-1] == '\t')) {
    --end;
  }
  return end > p && IsCompilerDirectiveSentinel(p, end - p) ? p : nullptr;
}

constexpr bool IsDirective(const char *match, const char *dir) {
  for (; *match; ++match) {
    if (*match != ToLowerCaseLetter(*dir++)) {
      return false;
    }
  }
  return true;
}

Prescanner::LineClassification Prescanner::ClassifyLine(
    const char *start) const {
  if (inFixedForm_) {
    if (std::optional<LineClassification> lc{
            IsFixedFormCompilerDirectiveLine(start)}) {
      return std::move(*lc);
    }
    if (IsFixedFormCommentLine(start)) {
      return {LineClassification::Kind::Comment};
    }
  } else {
    if (std::optional<LineClassification> lc{
            IsFreeFormCompilerDirectiveLine(start)}) {
      return std::move(*lc);
    }
    if (const char *bang{IsFreeFormComment(start)}) {
      return {LineClassification::Kind::Comment,
          static_cast<std::size_t>(bang - start)};
    }
  }
  if (std::optional<std::size_t> quoteOffset{IsIncludeLine(start)}) {
    return {LineClassification::Kind::IncludeLine, *quoteOffset};
  }
  if (const char *dir{IsPreprocessorDirectiveLine(start)}) {
    if (IsDirective("if", dir) || IsDirective("elif", dir) ||
        IsDirective("else", dir) || IsDirective("endif", dir)) {
      return {LineClassification::Kind::ConditionalCompilationDirective};
    } else if (IsDirective("include", dir)) {
      return {LineClassification::Kind::IncludeDirective};
    } else if (IsDirective("define", dir) || IsDirective("undef", dir)) {
      return {LineClassification::Kind::DefinitionDirective};
    } else {
      return {LineClassification::Kind::PreprocessorDirective};
    }
  }
  return {LineClassification::Kind::Source};
}

void Prescanner::SourceFormChange(std::string &&dir) {
  if (dir == "!dir$ free") {
    inFixedForm_ = false;
  } else if (dir == "!dir$ fixed") {
    inFixedForm_ = true;
  }
}

// Acquire and append compiler directive continuation lines to
// the tokens that constitute a compiler directive, even when those
// directive continuation lines are the result of macro expansion.
// (Not used when neither the original compiler directive line nor
// the directive continuation line result from preprocessing; regular
// line continuation during tokenization handles that normal case.)
bool Prescanner::CompilerDirectiveContinuation(
    TokenSequence &tokens, const char *origSentinel) {
  if (inFixedForm_ || tokens.empty() ||
      tokens.TokenAt(tokens.SizeInTokens() - 1) != "&") {
    return false;
  }
  LineClassification followingLine{ClassifyLine(nextLine_)};
  if (followingLine.kind == LineClassification::Kind::Comment) {
    nextLine_ += followingLine.payloadOffset; // advance to '!' or newline
    NextLine();
    return true;
  }
  CHECK(origSentinel != nullptr);
  directiveSentinel_ = origSentinel; // so IsDirective() is true
  const char *nextContinuation{
      followingLine.kind == LineClassification::Kind::CompilerDirective
          ? FreeFormContinuationLine(true)
          : nullptr};
  if (!nextContinuation &&
      followingLine.kind != LineClassification::Kind::Source) {
    return false;
  }
  auto origNextLine{nextLine_};
  BeginSourceLine(nextLine_);
  NextLine();
  TokenSequence followingTokens;
  if (nextContinuation) {
    // What follows is !DIR$ & xxx; skip over the & so that it
    // doesn't cause a spurious continuation.
    at_ = nextContinuation;
  } else {
    // What follows looks like a source line before macro expansion,
    // but might become a directive continuation afterwards.
    SkipSpaces();
  }
  while (NextToken(followingTokens)) {
  }
  if (auto followingPrepro{
          preprocessor_.MacroReplacement(followingTokens, *this)}) {
    followingTokens = std::move(*followingPrepro);
  }
  followingTokens.RemoveRedundantBlanks();
  std::size_t startAt{0};
  std::size_t keep{followingTokens.SizeInTokens()};
  bool ok{false};
  if (nextContinuation) {
    ok = true;
  } else {
    if (keep >= 3 && followingTokens.TokenAt(0) == "!" &&
        followingTokens.TokenAt(2) == "&") {
      CharBlock sentinel{followingTokens.TokenAt(1)};
      if (!sentinel.empty() &&
          std::memcmp(sentinel.begin(), origSentinel, sentinel.size()) == 0) {
        startAt = 3;
        keep -= 3;
        ok = true;
      }
    }
  }
  if (ok) {
    tokens.pop_back(); // delete original '&'
    tokens.Put(followingTokens, startAt, keep);
  } else {
    nextLine_ = origNextLine;
  }
  return ok;
}

// Similar, but for source line continuation after macro replacement.
bool Prescanner::SourceLineContinuation(TokenSequence &tokens) {
  if (!inFixedForm_ && !tokens.empty() &&
      tokens.TokenAt(tokens.SizeInTokens() - 1) == "&") {
    LineClassification followingLine{ClassifyLine(nextLine_)};
    if (followingLine.kind == LineClassification::Kind::Comment) {
      nextLine_ += followingLine.payloadOffset; // advance to '!' or newline
      NextLine();
      return true;
    } else if (const char *nextContinuation{FreeFormContinuationLine(true)}) {
      BeginSourceLine(nextLine_);
      NextLine();
      TokenSequence followingTokens;
      at_ = nextContinuation;
      while (NextToken(followingTokens)) {
      }
      if (auto followingPrepro{
              preprocessor_.MacroReplacement(followingTokens, *this)}) {
        followingTokens = std::move(*followingPrepro);
      }
      followingTokens.RemoveRedundantBlanks();
      tokens.pop_back(); // delete original '&'
      tokens.Put(followingTokens);
      return true;
    }
  }
  return false;
}
} // namespace Fortran::parser
