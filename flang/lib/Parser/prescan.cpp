//===-- lib/Parser/prescan.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "prescan.h"
#include "flang/Common/idioms.h"
#include "flang/Parser/characters.h"
#include "flang/Parser/message.h"
#include "flang/Parser/preprocessor.h"
#include "flang/Parser/source.h"
#include "flang/Parser/token-sequence.h"
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
      backslashFreeFormContinuation_{preprocessor.AnyDefinitions()},
      encoding_{allSources_.encoding()} {}

Prescanner::Prescanner(const Prescanner &that, Preprocessor &prepro,
    bool isNestedInIncludeDirective)
    : messages_{that.messages_}, cooked_{that.cooked_}, preprocessor_{prepro},
      allSources_{that.allSources_}, features_{that.features_},
      preprocessingOnly_{that.preprocessingOnly_},
      expandIncludeLines_{that.expandIncludeLines_},
      isNestedInIncludeDirective_{isNestedInIncludeDirective},
      backslashFreeFormContinuation_{that.backslashFreeFormContinuation_},
      inFixedForm_{that.inFixedForm_},
      fixedFormColumnLimit_{that.fixedFormColumnLimit_},
      encoding_{that.encoding_},
      prescannerNesting_{that.prescannerNesting_ + 1},
      skipLeadingAmpersand_{that.skipLeadingAmpersand_},
      compilerDirectiveBloomFilter_{that.compilerDirectiveBloomFilter_},
      compilerDirectiveSentinels_{that.compilerDirectiveSentinels_} {}

// Returns number of bytes to skip
static inline int IsSpace(const char *p) {
  if (*p == ' ') {
    return 1;
  } else if (*p == '\xa0') { // LATIN-1 NBSP non-breaking space
    return 1;
  } else if (p[0] == '\xc2' && p[1] == '\xa0') { // UTF-8 NBSP
    return 2;
  } else {
    return 0;
  }
}

static inline int IsSpaceOrTab(const char *p) {
  return *p == '\t' ? 1 : IsSpace(p);
}

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
  inFixedForm_ = beganInFixedForm;
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
    preprocessor_.Directive(TokenizePreprocessorDirective(), *this);
    afterPreprocessingDirective_ = true;
    skipLeadingAmpersand_ |= !inFixedForm_;
    return;
  case LineClassification::Kind::PreprocessorDirective:
    preprocessor_.Directive(TokenizePreprocessorDirective(), *this);
    afterPreprocessingDirective_ = true;
    // Don't set skipLeadingAmpersand_
    return;
  case LineClassification::Kind::DefinitionDirective:
    preprocessor_.Directive(TokenizePreprocessorDirective(), *this);
    // Don't set afterPreprocessingDirective_ or skipLeadingAmpersand_
    return;
  case LineClassification::Kind::CompilerDirective: {
    directiveSentinel_ = line.sentinel;
    CHECK(InCompilerDirective());
    BeginStatementAndAdvance();
    if (inFixedForm_) {
      CHECK(IsFixedFormCommentChar(*at_));
    } else {
      at_ += line.payloadOffset;
      column_ += line.payloadOffset;
      CHECK(*at_ == '!');
    }
    std::optional<int> condOffset;
    if (InOpenMPConditionalLine()) {
      condOffset = 2;
    } else if (directiveSentinel_[0] == '@' && directiveSentinel_[1] == 'c' &&
        directiveSentinel_[2] == 'u' && directiveSentinel_[3] == 'f' &&
        directiveSentinel_[4] == '\0') {
      // CUDA conditional compilation line.
      condOffset = 5;
    } else if (directiveSentinel_[0] == '@' && directiveSentinel_[1] == 'a' &&
        directiveSentinel_[2] == 'c' && directiveSentinel_[3] == 'c' &&
        directiveSentinel_[4] == '\0') {
      // OpenACC conditional compilation line.
      condOffset = 5;
    }
    if (condOffset && !preprocessingOnly_) {
      at_ += *condOffset, column_ += *condOffset;
      if (auto payload{IsIncludeLine(at_)}) {
        FortranInclude(at_ + *payload);
        return;
      }
      if (inFixedForm_) {
        LabelField(tokens);
      }
      SkipSpaces();
    } else {
      // Compiler directive.  Emit normalized sentinel, squash following spaces.
      // Conditional compilation lines (!$) take this path in -E mode too
      // so that -fopenmp only has to appear on the later compilation.
      EmitChar(tokens, '!');
      ++at_, ++column_;
      for (const char *sp{directiveSentinel_}; *sp != '\0';
           ++sp, ++at_, ++column_) {
        EmitChar(tokens, *sp);
      }
      if (inFixedForm_) {
        // We need to add the whitespace after the sentinel because otherwise
        // the line cannot be re-categorised as a compiler directive.
        while (column_ <= 6) {
          if (*at_ == '\t') {
            tabInCurrentLine_ = true;
            ++at_;
            for (; column_ < 7; ++column_) {
              EmitChar(tokens, ' ');
            }
          } else if (int spaceBytes{IsSpace(at_)}) {
            EmitChar(tokens, ' ');
            at_ += spaceBytes;
            ++column_;
          } else {
            if (InOpenMPConditionalLine() && column_ == 3 &&
                IsDecimalDigit(*at_)) {
              // subtle: !$ in -E mode can't be immediately followed by a digit
              EmitChar(tokens, ' ');
            }
            break;
          }
        }
      } else if (int spaceBytes{IsSpaceOrTab(at_)}) {
        EmitChar(tokens, ' ');
        at_ += spaceBytes, ++column_;
      }
      tokens.CloseToken();
      SkipSpaces();
      if (InOpenMPConditionalLine() && inFixedForm_ && !tabInCurrentLine_ &&
          column_ == 6 && *at_ != '\n') {
        // !$   0   - turn '0' into a space
        // !$   1   - turn '1' into '&'
        if (int n{IsSpace(at_)}; n || *at_ == '0') {
          at_ += n ? n : 1;
        } else {
          ++at_;
          EmitChar(tokens, '&');
          tokens.CloseToken();
        }
        ++column_;
        SkipSpaces();
      }
    }
    break;
  }
  case LineClassification::Kind::Source: {
    BeginStatementAndAdvance();
    bool checkLabelField{false};
    if (inFixedForm_) {
      if (features_.IsEnabled(LanguageFeature::OldDebugLines) &&
          (*at_ == 'D' || *at_ == 'd')) {
        NextChar();
      }
      checkLabelField = true;
    } else {
      if (skipLeadingAmpersand_) {
        skipLeadingAmpersand_ = false;
        const char *p{SkipWhiteSpace(at_)};
        if (p < limit_ && *p == '&') {
          column_ += ++p - at_;
          at_ = p;
        }
      } else {
        SkipSpaces();
      }
    }
    // Check for a leading identifier that might be a keyword macro
    // that will expand to anything indicating a non-source line, like
    // a comment marker or directive sentinel.  If so, disable line
    // continuation, so that NextToken() won't consume anything from
    // following lines.
    if (IsLegalIdentifierStart(*at_)) {
      // TODO: Only bother with these cases when any keyword macro has
      // been defined with replacement text that could begin a comment
      // or directive sentinel.
      const char *p{at_};
      while (IsLegalInIdentifier(*++p)) {
      }
      CharBlock id{at_, static_cast<std::size_t>(p - at_)};
      if (preprocessor_.IsNameDefined(id) &&
          !preprocessor_.IsFunctionLikeDefinition(id)) {
        checkLabelField = false;
        TokenSequence toks;
        toks.Put(id, GetProvenance(at_));
        if (auto replaced{preprocessor_.MacroReplacement(toks, *this)}) {
          auto newLineClass{ClassifyLine(*replaced, GetCurrentProvenance())};
          if (newLineClass.kind ==
              LineClassification::Kind::CompilerDirective) {
            directiveSentinel_ = newLineClass.sentinel;
            disableSourceContinuation_ = false;
          } else {
            disableSourceContinuation_ = !replaced->empty() &&
                newLineClass.kind != LineClassification::Kind::Source;
          }
        }
      }
    }
    if (checkLabelField) {
      LabelField(tokens);
    }
  } break;
  }

  while (NextToken(tokens)) {
  }
  if (continuationLines_ > 255) {
    if (features_.ShouldWarn(common::LanguageFeature::MiscSourceExtensions)) {
      Say(common::LanguageFeature::MiscSourceExtensions,
          GetProvenance(statementStart),
          "%d continuation lines is more than the Fortran standard allows"_port_en_US,
          continuationLines_);
    }
  }

  Provenance newlineProvenance{GetCurrentProvenance()};
  if (std::optional<TokenSequence> preprocessed{
          preprocessor_.MacroReplacement(tokens, *this)}) {
    // Reprocess the preprocessed line.
    LineClassification ppl{ClassifyLine(*preprocessed, newlineProvenance)};
    switch (ppl.kind) {
    case LineClassification::Kind::Comment:
      break;
    case LineClassification::Kind::IncludeLine:
      FortranInclude(preprocessed->TokenAt(0).begin() + ppl.payloadOffset);
      break;
    case LineClassification::Kind::ConditionalCompilationDirective:
    case LineClassification::Kind::IncludeDirective:
    case LineClassification::Kind::DefinitionDirective:
    case LineClassification::Kind::PreprocessorDirective:
      if (features_.ShouldWarn(common::UsageWarning::Preprocessing)) {
        Say(common::UsageWarning::Preprocessing,
            preprocessed->GetProvenanceRange(),
            "Preprocessed line resembles a preprocessor directive"_warn_en_US);
      }
      CheckAndEmitLine(preprocessed->ToLowerCase(), newlineProvenance);
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
      if (!SourceFormChange(preprocessed->ToString())) {
        CheckAndEmitLine(
            preprocessed->ClipComment(*this, true /* skip first ! */),
            newlineProvenance);
      }
      break;
    case LineClassification::Kind::Source:
      if (inFixedForm_) {
        if (!preprocessingOnly_ && preprocessed->HasBlanks()) {
          preprocessed->RemoveBlanks();
        }
      } else {
        while (SourceLineContinuation(*preprocessed)) {
          newlineProvenance = GetCurrentProvenance();
        }
        if (preprocessed->HasRedundantBlanks()) {
          preprocessed->RemoveRedundantBlanks();
        }
      }
      CheckAndEmitLine(
          preprocessed->ToLowerCase().ClipComment(*this), newlineProvenance);
      break;
    }
  } else { // no macro replacement
    if (line.kind == LineClassification::Kind::CompilerDirective) {
      while (CompilerDirectiveContinuation(tokens, line.sentinel)) {
        newlineProvenance = GetCurrentProvenance();
      }
      if (preprocessingOnly_ && inFixedForm_ && InOpenMPConditionalLine() &&
          nextLine_ < limit_) {
        // In -E mode, when the line after !$ conditional compilation is a
        // regular fixed form continuation line, append a '&' to the line.
        const char *p{nextLine_};
        int col{1};
        while (int n{IsSpace(p)}) {
          if (*p == '\t') {
            break;
          }
          p += n;
          ++col;
        }
        if (col == 6 && *p != '0' && *p != '\t' && *p != '\n') {
          EmitChar(tokens, '&');
          tokens.CloseToken();
        }
      }
      tokens.ToLowerCase();
      if (!SourceFormChange(tokens.ToString())) {
        CheckAndEmitLine(tokens, newlineProvenance);
      }
    } else { // Kind::Source
      tokens.ToLowerCase();
      if (inFixedForm_) {
        EnforceStupidEndStatementRules(tokens);
      }
      CheckAndEmitLine(tokens, newlineProvenance);
    }
  }
  directiveSentinel_ = nullptr;
}

void Prescanner::CheckAndEmitLine(
    TokenSequence &tokens, Provenance newlineProvenance) {
  tokens.CheckBadFortranCharacters(
      messages_, *this, disableSourceContinuation_ || preprocessingOnly_);
  // Parenthesis nesting check does not apply while any #include is
  // active, nor on the lines before and after a top-level #include,
  // nor before or after conditional source.
  // Applications play shenanigans with line continuation before and
  // after #include'd subprogram argument lists and conditional source.
  if (!preprocessingOnly_ && !isNestedInIncludeDirective_ && !omitNewline_ &&
      !afterPreprocessingDirective_ && tokens.BadlyNestedParentheses() &&
      !preprocessor_.InConditional()) {
    if (nextLine_ < limit_ && IsPreprocessorDirectiveLine(nextLine_)) {
      // don't complain
    } else {
      tokens.CheckBadParentheses(messages_);
    }
  }
  tokens.Emit(cooked_);
  if (omitNewline_) {
    omitNewline_ = false;
  } else {
    cooked_.Put('\n', newlineProvenance);
    afterPreprocessingDirective_ = false;
  }
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
    if (int n{IsSpace(at_)}; n == 0 &&
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
    if ((prescannerNesting_ > 0 && *badColumn == 6 &&
            cooked_.BufferedBytes() == firstCookedCharacterOffset_) ||
        afterPreprocessingDirective_) {
      // This is the first source line in #include'd text or conditional
      // code under #if, or the first source line after such.
      // If it turns out that the preprocessed text begins with a
      // fixed form continuation line, the newline at the end
      // of the latest source line beforehand will be deleted in
      // CookedSource::Marshal().
      cooked_.MarkPossibleFixedFormContinuation();
    } else if (features_.ShouldWarn(common::UsageWarning::Scanning)) {
      Say(common::UsageWarning::Scanning, GetProvenance(start + *badColumn - 1),
          *badColumn == 6
              ? "Statement should not begin with a continuation line"_warn_en_US
              : "Character in fixed-form label field must be a digit"_warn_en_US);
    }
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
      Say(common::LanguageFeature::MiscSourceExtensions, GetCurrentProvenance(),
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
  } else if (*at_ == '!' && !inCharLiteral_ &&
      (!inFixedForm_ || tabInCurrentLine_ || column_ != 6)) {
    return !IsCompilerDirectiveSentinel(at_);
  } else {
    return false;
  }
}

void Prescanner::NextChar() {
  CHECK(*at_ != '\n');
  int n{IsSpace(at_)};
  at_ += n ? n : 1;
  ++column_;
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
bool Prescanner::SkipToNextSignificantCharacter() {
  if (inPreprocessorDirective_) {
    SkipCComments();
    return false;
  } else {
    auto anyContinuationLine{false};
    bool atNewline{false};
    if (MustSkipToEndOfLine()) {
      SkipToEndOfLine();
    } else {
      atNewline = *at_ == '\n';
    }
    for (; Continuation(atNewline); atNewline = false) {
      anyContinuationLine = true;
      ++continuationLines_;
      if (MustSkipToEndOfLine()) {
        SkipToEndOfLine();
      }
    }
    if (*at_ == '\t') {
      tabInCurrentLine_ = true;
    }
    return anyContinuationLine;
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
  while (IsSpaceOrTab(at_)) {
    NextChar();
  }
  brokenToken_ = false;
}

const char *Prescanner::SkipWhiteSpace(const char *p) {
  while (int n{IsSpaceOrTab(p)}) {
    p += n;
  }
  return p;
}

const char *Prescanner::SkipWhiteSpaceIncludingEmptyMacros(
    const char *p) const {
  while (true) {
    if (int n{IsSpaceOrTab(p)}) {
      p += n;
    } else if (preprocessor_.AnyDefinitions() && IsLegalIdentifierStart(*p)) {
      // Skip keyword macros with empty definitions
      const char *q{p + 1};
      while (IsLegalInIdentifier(*q)) {
        ++q;
      }
      if (preprocessor_.IsNameDefinedEmpty(
              CharBlock{p, static_cast<std::size_t>(q - p)})) {
        p = q;
      } else {
        break;
      }
    } else {
      break;
    }
  }
  return p;
}

const char *Prescanner::SkipWhiteSpaceAndCComments(const char *p) const {
  while (true) {
    if (int n{IsSpaceOrTab(p)}) {
      p += n;
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
  if (InFixedFormSource() && !preprocessingOnly_) {
    SkipSpaces();
  } else {
    if (*at_ == '/' && IsCComment(at_)) {
      // Recognize and skip over classic C style /*comments*/ when
      // outside a character literal.
      if (features_.ShouldWarn(LanguageFeature::ClassicCComments)) {
        Say(LanguageFeature::ClassicCComments, GetCurrentProvenance(),
            "nonstandard usage: C-style comment"_port_en_US);
      }
      SkipCComments();
    }
    if (IsSpaceOrTab(at_)) {
      // Compress free-form white space into a single space character.
      const auto theSpace{at_};
      char previous{at_ <= start_ ? ' ' : at_[-1]};
      NextChar();
      SkipSpaces();
      if (*at_ == '\n' && !omitNewline_) {
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
  brokenToken_ = false;
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
      HandleExponentAndOrKindSuffix(tokens);
    } else if (HandleExponentAndOrKindSuffix(tokens)) {
    } else if (digits == 1 && n == 0 && (*at_ == 'x' || *at_ == 'X') &&
        inPreprocessorDirective_) {
      do {
        EmitCharAndAdvance(tokens, *at_);
      } while (IsHexadecimalDigit(*at_));
    } else if (at_[0] == '_' && (at_[1] == '\'' || at_[1] == '"')) { // 4_"..."
      EmitCharAndAdvance(tokens, *at_);
      QuotedCharacterLiteral(tokens, start);
    } else if (IsLetter(*at_) && !preventHollerith_ &&
        parenthesisNesting_ > 0 &&
        !preprocessor_.IsNameDefined(CharBlock{at_, 1})) {
      // Handles FORMAT(3I9HHOLLERITH) by skipping over the first I so that
      // we don't misrecognize I9HHOLLERITH as an identifier in the next case.
      EmitCharAndAdvance(tokens, *at_);
    }
    preventHollerith_ = false;
  } else if (*at_ == '.') {
    char nch{EmitCharAndAdvance(tokens, '.')};
    if (!inPreprocessorDirective_ && IsDecimalDigit(nch)) {
      while (IsDecimalDigit(EmitCharAndAdvance(tokens, *at_))) {
      }
      HandleExponentAndOrKindSuffix(tokens);
    } else if (nch == '.' && EmitCharAndAdvance(tokens, '.') == '.') {
      EmitCharAndAdvance(tokens, '.'); // variadic macro definition ellipsis
    }
    preventHollerith_ = false;
  } else if (IsLegalInIdentifier(*at_)) {
    std::size_t parts{1};
    bool anyDefined{false};
    bool hadContinuation{false};
    // Subtlety: When an identifier is split across continuation lines,
    // its parts are kept as distinct pp-tokens if macro replacement
    // should operate on them independently.  This trick accommodates the
    // historic practice of using line continuation for token pasting after
    // replacement.
    // In free form, the macro to be replaced must have been preceded
    // by '&' and followed by either '&' or, if last, the end of a line.
    //   call &                call foo&        call foo&
    //     &MACRO&      OR       &MACRO&   OR     &MACRO
    //     &foo(...)             &(...)
    do {
      EmitChar(tokens, *at_);
      ++at_, ++column_;
      hadContinuation = SkipToNextSignificantCharacter();
      if (hadContinuation && IsLegalIdentifierStart(*at_)) {
        if (brokenToken_) {
          break;
        }
        // Continued identifier
        tokens.CloseToken();
        ++parts;
        if (!anyDefined &&
            (parts > 2 || inFixedForm_ ||
                (start > start_ && start[-1] == '&')) &&
            preprocessor_.IsNameDefined(
                tokens.TokenAt(tokens.SizeInTokens() - 1))) {
          anyDefined = true;
        }
      }
    } while (IsLegalInIdentifier(*at_));
    if (!anyDefined && parts > 1) {
      tokens.CloseToken();
      char after{*SkipWhiteSpace(at_)};
      anyDefined = (hadContinuation || after == '\n' || after == '&') &&
          preprocessor_.IsNameDefined(
              tokens.TokenAt(tokens.SizeInTokens() - 1));
      tokens.ReopenLastToken();
    }
    if (!anyDefined) {
      // If no part was a defined macro, combine the parts into one so that
      // the combination itself can be subject to macro replacement.
      while (parts-- > 1) {
        tokens.ReopenLastToken();
      }
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
          Say(common::LanguageFeature::MiscSourceExtensions,
              GetProvenanceRange(at_, at_ + 1),
              "Label should be in the label field"_port_en_US);
        }
      }
    }
  }
  tokens.CloseToken();
  return true;
}

bool Prescanner::HandleExponent(TokenSequence &tokens) {
  if (char ed{ToLowerCaseLetter(*at_)}; ed == 'e' || ed == 'd') {
    // Do some look-ahead to ensure that this 'e'/'d' is an exponent,
    // not the start of an identifier that could be a macro.
    const char *startAt{at_};
    int startColumn{column_};
    TokenSequence possible;
    EmitCharAndAdvance(possible, *at_);
    if (InFixedFormSource()) {
      SkipSpaces();
    }
    if (*at_ == '+' || *at_ == '-') {
      EmitCharAndAdvance(possible, *at_);
      if (InFixedFormSource()) {
        SkipSpaces();
      }
    }
    if (IsDecimalDigit(*at_)) { // it's an exponent; scan it
      while (IsDecimalDigit(*at_)) {
        EmitCharAndAdvance(possible, *at_);
        if (InFixedFormSource()) {
          SkipSpaces();
        }
      }
      possible.CloseToken();
      tokens.AppendRange(possible, 0); // appends to current token
      return true;
    }
    // Not an exponent; backtrack
    at_ = startAt;
    column_ = startColumn;
  }
  return false;
}

bool Prescanner::HandleKindSuffix(TokenSequence &tokens) {
  if (*at_ != '_') {
    return false;
  }
  TokenSequence withUnderscore, separate;
  EmitChar(withUnderscore, '_');
  EmitCharAndAdvance(separate, '_');
  if (InFixedFormSource()) {
    SkipSpaces();
  }
  if (IsLegalInIdentifier(*at_)) {
    separate.CloseToken();
    EmitChar(withUnderscore, *at_);
    EmitCharAndAdvance(separate, *at_);
    if (InFixedFormSource()) {
      SkipSpaces();
    }
    while (IsLegalInIdentifier(*at_)) {
      EmitChar(withUnderscore, *at_);
      EmitCharAndAdvance(separate, *at_);
      if (InFixedFormSource()) {
        SkipSpaces();
      }
    }
  }
  withUnderscore.CloseToken();
  separate.CloseToken();
  tokens.CloseToken();
  if (separate.SizeInTokens() == 2 &&
      preprocessor_.IsNameDefined(separate.TokenAt(1)) &&
      !preprocessor_.IsNameDefined(withUnderscore.ToCharBlock())) {
    // "_foo" is not defined, but "foo" is
    tokens.CopyAll(separate); // '_' "foo"
  } else {
    tokens.CopyAll(withUnderscore); // "_foo"
  }
  return true;
}

bool Prescanner::HandleExponentAndOrKindSuffix(TokenSequence &tokens) {
  bool hadExponent{HandleExponent(tokens)};
  if (HandleKindSuffix(tokens)) {
    return true;
  } else {
    return hadExponent;
  }
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
    if (*at_ == '\n') {
      if (inPreprocessorDirective_) {
        EmitQuotedChar(static_cast<unsigned char>(*at_), emit, insert, false,
            Encoding::LATIN_1);
      } else if (InCompilerDirective() && preprocessingOnly_) {
        // don't complain about -E output of !$, do it in later compilation
      } else {
        Say(GetProvenanceRange(start, end),
            "Incomplete character literal"_err_en_US);
      }
      break;
    }
    EmitQuotedChar(static_cast<unsigned char>(*at_), emit, insert, false,
        Encoding::LATIN_1);
    while (PadOutCharacterLiteral(tokens)) {
    }
    // Here's a weird edge case.  When there's a two or more following
    // continuation lines at this point, and the entire significant part of
    // the next continuation line is the name of a keyword macro, replace
    // it in the character literal with its definition.  Example:
    //   #define FOO foo
    //   subroutine subr() bind(c, name="my_&
    //     &FOO&
    //     &_bar") ...
    // produces a binding name of "my_foo_bar".
    while (at_[1] == '&' && nextLine_ < limit_ && !InFixedFormSource()) {
      const char *idStart{nextLine_};
      if (const char *amper{SkipWhiteSpace(nextLine_)}; *amper == '&') {
        idStart = amper + 1;
      }
      if (IsLegalIdentifierStart(*idStart)) {
        std::size_t idLen{1};
        for (; IsLegalInIdentifier(idStart[idLen]); ++idLen) {
        }
        if (idStart[idLen] == '&') {
          CharBlock id{idStart, idLen};
          if (preprocessor_.IsNameDefined(id)) {
            TokenSequence ppTokens;
            ppTokens.Put(id, GetProvenance(idStart));
            if (auto replaced{
                    preprocessor_.MacroReplacement(ppTokens, *this)}) {
              tokens.CopyAll(*replaced);
              at_ = &idStart[idLen - 1];
              NextLine();
              continue; // try again on the next line
            }
          }
        }
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
      if (features_.ShouldWarn(common::UsageWarning::Scanning)) {
        Say(common::UsageWarning::Scanning, GetProvenanceRange(start, at_),
            "Possible truncated Hollerith literal"_warn_en_US);
      }
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
    if (int n{IsSpace(p)}) {
      p += n;
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
  if (!expandIncludeLines_) {
    return std::nullopt;
  }
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
    if (features_.ShouldWarn(common::UsageWarning::Scanning)) {
      Say(common::UsageWarning::Scanning, GetProvenanceRange(garbage, p),
          "excess characters after path name"_warn_en_US);
    }
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
    Say(provenance, "INCLUDE: %s"_err_en_US, buf);
  } else if (included->bytes() > 0) {
    ProvenanceRange includeLineRange{
        provenance, static_cast<std::size_t>(p - nextLine_)};
    ProvenanceRange fileRange{
        allSources_.AddIncludedFile(*included, includeLineRange)};
    Preprocessor cleanPrepro{allSources_};
    if (preprocessor_.IsNameDefined("__FILE__"s)) {
      cleanPrepro.DefineStandardMacros(); // __FILE__, __LINE__, &c.
    }
    if (preprocessor_.IsNameDefined("_CUDA"s)) {
      cleanPrepro.Define("_CUDA"s, "1");
    }
    Prescanner{*this, cleanPrepro, /*isNestedInIncludeDirective=*/false}
        .set_encoding(included->encoding())
        .Prescan(fileRange);
  }
}

const char *Prescanner::IsPreprocessorDirectiveLine(const char *start) const {
  const char *p{start};
  while (int n{IsSpace(p)}) {
    p += n;
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
  } else if (inPreprocessorDirective_) {
  } else {
    auto lineClass{ClassifyLine(nextLine_)};
    if (lineClass.kind == LineClassification::Kind::Comment) {
      NextLine();
      return true;
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
        (lineClass.kind == LineClassification::Kind::DefinitionDirective ||
            lineClass.kind == LineClassification::Kind::IncludeDirective ||
            lineClass.kind == LineClassification::Kind::IncludeLine)) {
      SkipToEndOfLine();
      omitNewline_ = true;
      skipLeadingAmpersand_ = true;
    }
  }
  return false;
}

const char *Prescanner::FixedFormContinuationLine(bool atNewline) {
  if (IsAtEnd()) {
    return nullptr;
  }
  tabInCurrentLine_ = false;
  char col1{*nextLine_};
  bool canBeNonDirectiveContinuation{
      (col1 == ' ' ||
          ((col1 == 'D' || col1 == 'd') &&
              features_.IsEnabled(LanguageFeature::OldDebugLines))) &&
      nextLine_[1] == ' ' && nextLine_[2] == ' ' && nextLine_[3] == ' ' &&
      nextLine_[4] == ' '};
  if (InCompilerDirective() &&
      !(InOpenMPConditionalLine() && !preprocessingOnly_)) {
    // !$ under -E is not continued, but deferred to later compilation
    if (IsFixedFormCommentChar(col1) &&
        !(InOpenMPConditionalLine() && preprocessingOnly_)) {
      int j{1};
      for (; j < 5; ++j) {
        char ch{directiveSentinel_[j - 1]};
        if (ch == '\0') {
          break;
        } else if (ch != ToLowerCaseLetter(nextLine_[j])) {
          return nullptr;
        }
      }
      for (; j < 5; ++j) {
        if (nextLine_[j] != ' ') {
          return nullptr;
        }
      }
      const char *col6{nextLine_ + 5};
      if (*col6 != '\n' && *col6 != '0' && !IsSpaceOrTab(col6)) {
        if (atNewline && !IsSpace(nextLine_ + 6)) {
          brokenToken_ = true;
        }
        return nextLine_ + 6;
      }
    }
  } else { // Normal case: not in a compiler directive.
    // !$ conditional compilation lines may be continuations when not
    // just preprocessing.
    if (!preprocessingOnly_ && IsFixedFormCommentChar(col1) &&
        nextLine_[1] == '$' && nextLine_[2] == ' ' && nextLine_[3] == ' ' &&
        nextLine_[4] == ' ' && IsCompilerDirectiveSentinel(&nextLine_[1], 1)) {
      if (const char *col6{nextLine_ + 5};
          *col6 != '\n' && *col6 != '0' && !IsSpaceOrTab(col6)) {
        if (atNewline && !IsSpace(nextLine_ + 6)) {
          brokenToken_ = true;
        }
        return nextLine_ + 6;
      } else {
        return nullptr;
      }
    }
    if (col1 == '&' &&
        features_.IsEnabled(
            LanguageFeature::FixedFormContinuationWithColumn1Ampersand)) {
      // Extension: '&' as continuation marker
      if (features_.ShouldWarn(
              LanguageFeature::FixedFormContinuationWithColumn1Ampersand)) {
        Say(LanguageFeature::FixedFormContinuationWithColumn1Ampersand,
            GetProvenance(nextLine_), "nonstandard usage"_port_en_US);
      }
      return nextLine_ + 1;
    }
    if (col1 == '\t' && nextLine_[1] >= '1' && nextLine_[1] <= '9') {
      tabInCurrentLine_ = true;
      return nextLine_ + 2; // VAX extension
    }
    if (canBeNonDirectiveContinuation) {
      const char *col6{nextLine_ + 5};
      if (*col6 != '\n' && *col6 != '0' && !IsSpaceOrTab(col6)) {
        if ((*col6 == 'i' || *col6 == 'I') && IsIncludeLine(nextLine_)) {
          // It's an INCLUDE line, not a continuation
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
  const char *lineStart{nextLine_};
  const char *p{lineStart};
  if (p >= limit_) {
    return nullptr;
  }
  p = SkipWhiteSpaceIncludingEmptyMacros(p);
  if (InCompilerDirective()) {
    if (InOpenMPConditionalLine()) {
      if (preprocessingOnly_) {
        // in -E mode, don't treat !$ as a continuation
        return nullptr;
      } else if (p[0] == '!' && p[1] == '$') {
        // accept but do not require a matching sentinel
        if (p[2] != '&' && !IsSpaceOrTab(&p[2])) {
          return nullptr; // not !$
        }
        p += 2;
      }
    } else if (*p++ == '!') {
      for (const char *s{directiveSentinel_}; *s != '\0'; ++p, ++s) {
        if (*s != ToLowerCaseLetter(*p)) {
          return nullptr; // not the same directive class
        }
      }
    } else {
      return nullptr;
    }
    p = SkipWhiteSpace(p);
    if (*p == '&') {
      if (!ampersand) {
        brokenToken_ = true;
      }
      return p + 1;
    } else if (ampersand) {
      return p;
    } else {
      return nullptr;
    }
  }
  if (p[0] == '!' && p[1] == '$' && !preprocessingOnly_ &&
      features_.IsEnabled(LanguageFeature::OpenMP)) {
    // !$ conditional line can be a continuation
    p = lineStart = SkipWhiteSpace(p + 2);
  }
  if (*p == '&') {
    return p + 1;
  } else if (*p == '!' || *p == '\n' || *p == '#') {
    return nullptr;
  } else if (ampersand || IsImplicitContinuation()) {
    if (continuationInCharLiteral_) {
      // 'a'&            -> 'a''b' == "a'b"
      //   'b'
      if (features_.ShouldWarn(common::LanguageFeature::MiscSourceExtensions)) {
        Say(common::LanguageFeature::MiscSourceExtensions,
            GetProvenanceRange(p, p + 1),
            "Character literal continuation line should have been preceded by '&'"_port_en_US);
      }
    } else if (p > lineStart && IsSpaceOrTab(p - 1)) {
      --p;
    } else {
      brokenToken_ = true;
    }
    return p;
  } else {
    return nullptr;
  }
}

bool Prescanner::FixedFormContinuation(bool atNewline) {
  // N.B. We accept '&' as a continuation indicator in fixed form, too,
  // but not in a character literal.
  if (*at_ == '&' && inCharLiteral_) {
    return false;
  }
  do {
    if (const char *cont{FixedFormContinuationLine(atNewline)}) {
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
    } else if (ampersand && preprocessingOnly_ && !parenthesisNesting_) {
      return false; // allow & at start of line, maybe after !$
    } else if (features_.ShouldWarn(LanguageFeature::CruftAfterAmpersand)) {
      Say(LanguageFeature::CruftAfterAmpersand, GetProvenance(p),
          "missing ! before comment after &"_warn_en_US);
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
  if (disableSourceContinuation_) {
    return false;
  } else if (*at_ == '\n' || *at_ == '&') {
    if (inFixedForm_) {
      return FixedFormContinuation(mightNeedFixedFormSpace);
    } else {
      return FreeFormContinuation();
    }
  } else if (*at_ == '\\' && at_ + 2 == nextLine_ &&
      backslashFreeFormContinuation_ && !inFixedForm_ && nextLine_ < limit_) {
    // cpp-like handling of \ at end of a free form source line
    BeginSourceLine(nextLine_);
    NextLine();
    return true;
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
  for (; column < 6; ++column) {
    if (*p == '\n' || IsSpaceOrTab(p) || IsDecimalDigit(*p)) {
      break;
    }
    *sp++ = ToLowerCaseLetter(*p++);
  }
  if (sp == sentinel) {
    return std::nullopt;
  }
  *sp = '\0';
  // A fixed form OpenMP conditional compilation sentinel must satisfy the
  // following criteria, for initial lines:
  // - Columns 3 through 5 must have only white space or numbers.
  // - Column 6 must be space or zero.
  bool isOpenMPConditional{sp == &sentinel[1] && sentinel[0] == '$'};
  bool hadDigit{false};
  if (isOpenMPConditional) {
    for (; column < 6; ++column, ++p) {
      if (IsDecimalDigit(*p)) {
        hadDigit = true;
      } else if (!IsSpaceOrTab(p)) {
        return std::nullopt;
      }
    }
  }
  if (column == 6) {
    if (*p == '0') {
      ++p;
    } else if (int n{IsSpaceOrTab(p)}) {
      p += n;
    } else if (isOpenMPConditional && preprocessingOnly_ && !hadDigit &&
        *p != '\n') {
      // In -E mode, "!$   &" is treated as a directive
    } else {
      // This is a Continuation line, not an initial directive line.
      return std::nullopt;
    }
  }
  if (const char *ss{IsCompilerDirectiveSentinel(
          sentinel, static_cast<std::size_t>(sp - sentinel))}) {
    return {
        LineClassification{LineClassification::Kind::CompilerDirective, 0, ss}};
  }
  return std::nullopt;
}

std::optional<Prescanner::LineClassification>
Prescanner::IsFreeFormCompilerDirectiveLine(const char *start) const {
  if (const char *p{SkipWhiteSpaceIncludingEmptyMacros(start)};
      p && *p++ == '!') {
    if (auto maybePair{IsCompilerDirectiveSentinel(p)}) {
      auto offset{static_cast<std::size_t>(p - start - 1)};
      return {LineClassification{LineClassification::Kind::CompilerDirective,
          offset, maybePair->first}};
    }
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

std::optional<std::pair<const char *, const char *>>
Prescanner::IsCompilerDirectiveSentinel(const char *p) const {
  char sentinel[8];
  for (std::size_t j{0}; j + 1 < sizeof sentinel; ++p, ++j) {
    if (int n{IsSpaceOrTab(p)};
        n || !(IsLetter(*p) || *p == '$' || *p == '@')) {
      if (j > 0) {
        if (j == 1 && sentinel[0] == '$' && n == 0 && *p != '&' && *p != '\n') {
          // Free form OpenMP conditional compilation line sentinels have to
          // be immediately followed by a space or &, not a digit
          // or anything else.  A newline also works for an initial line.
          break;
        }
        sentinel[j] = '\0';
        if (*p != '!') {
          if (const char *sp{IsCompilerDirectiveSentinel(sentinel, j)}) {
            return std::make_pair(sp, p);
          }
        }
      }
      break;
    } else {
      sentinel[j] = ToLowerCaseLetter(*p);
    }
  }
  return std::nullopt;
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

Prescanner::LineClassification Prescanner::ClassifyLine(
    TokenSequence &tokens, Provenance newlineProvenance) const {
  // Append a newline temporarily.
  tokens.PutNextTokenChar('\n', newlineProvenance);
  tokens.CloseToken();
  const char *ppd{tokens.ToCharBlock().begin()};
  LineClassification classification{ClassifyLine(ppd)};
  tokens.pop_back(); // remove the newline
  return classification;
}

bool Prescanner::SourceFormChange(std::string &&dir) {
  if (dir == "!dir$ free") {
    inFixedForm_ = false;
    return true;
  } else if (dir == "!dir$ fixed") {
    inFixedForm_ = true;
    return true;
  } else {
    return false;
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
      tokens.TokenAt(tokens.SizeInTokens() - 1) != "&" ||
      (preprocessingOnly_ && !parenthesisNesting_)) {
    return false;
  }
  LineClassification followingLine{ClassifyLine(nextLine_)};
  if (followingLine.kind == LineClassification::Kind::Comment) {
    nextLine_ += followingLine.payloadOffset; // advance to '!' or newline
    NextLine();
    return true;
  }
  CHECK(origSentinel != nullptr);
  directiveSentinel_ = origSentinel; // so InCompilerDirective() is true
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
  if (nextContinuation) {
    // What follows is !DIR$ & xxx; skip over the & so that it
    // doesn't cause a spurious continuation.
    at_ = nextContinuation;
  } else {
    // What follows looks like a source line before macro expansion,
    // but might become a directive continuation afterwards.
    SkipSpaces();
  }
  TokenSequence followingTokens;
  while (NextToken(followingTokens)) {
  }
  if (auto followingPrepro{
          preprocessor_.MacroReplacement(followingTokens, *this)}) {
    followingTokens = std::move(*followingPrepro);
  }
  followingTokens.RemoveRedundantBlanks();
  std::size_t startAt{0};
  std::size_t following{followingTokens.SizeInTokens()};
  bool ok{false};
  if (nextContinuation) {
    ok = true;
  } else {
    startAt = 2;
    if (startAt < following && followingTokens.TokenAt(0) == "!") {
      CharBlock sentinel{followingTokens.TokenAt(1)};
      if (!sentinel.empty() &&
          std::memcmp(sentinel.begin(), origSentinel, sentinel.size()) == 0) {
        ok = true;
        while (
            startAt < following && followingTokens.TokenAt(startAt).IsBlank()) {
          ++startAt;
        }
        if (startAt < following && followingTokens.TokenAt(startAt) == "&") {
          ++startAt;
        }
      }
    }
  }
  if (ok) {
    tokens.pop_back(); // delete original '&'
    tokens.AppendRange(followingTokens, startAt, following - startAt);
    tokens.RemoveRedundantBlanks();
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
      tokens.CopyAll(followingTokens);
      return true;
    }
  }
  return false;
}
} // namespace Fortran::parser
