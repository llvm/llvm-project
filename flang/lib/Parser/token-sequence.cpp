//===-- lib/Parser/token-sequence.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Parser/token-sequence.h"

#include "prescan.h"
#include "flang/Parser/characters.h"
#include "flang/Parser/message.h"
#include "llvm/Support/raw_ostream.h"

namespace Fortran::parser {

TokenSequence &TokenSequence::operator=(TokenSequence &&that) {
  clear();
  swap(that);
  return *this;
}

void TokenSequence::clear() {
  start_.clear();
  nextStart_ = 0;
  char_.clear();
  provenances_.clear();
}

void TokenSequence::pop_back() {
  CHECK(!start_.empty());
  // If the last token is empty then `nextStart_ == start_.back()`.
  CHECK(nextStart_ >= start_.back());
  std::size_t bytes{nextStart_ - start_.back()};
  nextStart_ = start_.back();
  start_.pop_back();
  char_.resize(nextStart_);
  provenances_.RemoveLastBytes(bytes);
}

void TokenSequence::shrink_to_fit() {
  start_.shrink_to_fit();
  char_.shrink_to_fit();
  provenances_.shrink_to_fit();
}

void TokenSequence::swap(TokenSequence &that) {
  start_.swap(that.start_);
  std::swap(nextStart_, that.nextStart_);
  char_.swap(that.char_);
  provenances_.swap(that.provenances_);
}

std::size_t TokenSequence::SkipBlanks(std::size_t at) const {
  std::size_t tokens{start_.size()};
  for (; at < tokens; ++at) {
    if (!TokenAt(at).IsBlank()) {
      return at;
    }
  }
  return tokens; // even if at > tokens
}

std::optional<std::size_t> TokenSequence::SkipBlanksBackwards(
    std::size_t at) const {
  while (at-- > 0) {
    if (!TokenAt(at).IsBlank()) {
      return at;
    }
  }
  return std::nullopt;
}

// C-style /*comments*/ are removed from preprocessing directive
// token sequences by the prescanner, but not C++ or Fortran
// free-form line-ending comments (//...  and !...) because
// ignoring them is directive-specific.
bool TokenSequence::IsAnythingLeft(std::size_t at) const {
  std::size_t tokens{start_.size()};
  for (; at < tokens; ++at) {
    auto tok{TokenAt(at)};
    const char *end{tok.end()};
    for (const char *p{tok.begin()}; p < end; ++p) {
      switch (*p) {
      case '/':
        return p + 1 >= end || p[1] != '/';
      case '!':
        return false;
      case ' ':
        break;
      default:
        return true;
      }
    }
  }
  return false;
}

void TokenSequence::CopyAll(const TokenSequence &that) {
  if (nextStart_ < char_.size()) {
    start_.push_back(nextStart_);
  }
  int offset = char_.size();
  for (int st : that.start_) {
    start_.push_back(st + offset);
  }
  char_.insert(char_.end(), that.char_.begin(), that.char_.end());
  nextStart_ = char_.size();
  provenances_.Put(that.provenances_);
}

void TokenSequence::CopyWithProvenance(
    const TokenSequence &that, ProvenanceRange range) {
  std::size_t offset{0};
  std::size_t tokens{that.SizeInTokens()};
  for (std::size_t j{0}; j < tokens; ++j) {
    CharBlock tok{that.TokenAt(j)};
    Put(tok, range.OffsetMember(offset));
    offset += tok.size();
  }
  CHECK(offset == range.size());
}

void TokenSequence::AppendRange(
    const TokenSequence &that, std::size_t at, std::size_t tokens) {
  ProvenanceRange provenance;
  std::size_t offset{0};
  for (; tokens-- > 0; ++at) {
    CharBlock tok{that.TokenAt(at)};
    std::size_t tokBytes{tok.size()};
    for (std::size_t j{0}; j < tokBytes; ++j) {
      if (offset == provenance.size()) {
        provenance = that.provenances_.Map(that.start_[at] + j);
        offset = 0;
      }
      PutNextTokenChar(tok[j], provenance.OffsetMember(offset++));
    }
    CloseToken();
  }
}

void TokenSequence::Put(
    const char *s, std::size_t bytes, Provenance provenance) {
  for (std::size_t j{0}; j < bytes; ++j) {
    PutNextTokenChar(s[j], provenance + j);
  }
  CloseToken();
}

void TokenSequence::Put(const CharBlock &t, Provenance provenance) {
  // Avoid t[0] if t is empty: it would create a reference to nullptr,
  // which is UB.
  const char *addr{t.size() ? &t[0] : nullptr};
  Put(addr, t.size(), provenance);
}

void TokenSequence::Put(const std::string &s, Provenance provenance) {
  Put(s.data(), s.size(), provenance);
}

void TokenSequence::Put(llvm::raw_string_ostream &ss, Provenance provenance) {
  Put(ss.str(), provenance);
}

TokenSequence &TokenSequence::ToLowerCase() {
  std::size_t tokens{start_.size()};
  std::size_t chars{char_.size()};
  std::size_t atToken{0};
  for (std::size_t j{0}; j < chars;) {
    std::size_t nextStart{atToken + 1 < tokens ? start_[++atToken] : chars};
    char *p{&char_[j]};
    char const *limit{char_.data() + nextStart};
    const char *lastChar{limit - 1};
    j = nextStart;
    // Skip leading whitespaces
    while (p < limit - 1 && *p == ' ') {
      ++p;
    }
    // Find last non-whitespace char
    while (lastChar > p + 1 && *lastChar == ' ') {
      --lastChar;
    }
    if (IsDecimalDigit(*p)) {
      while (p < limit && IsDecimalDigit(*p)) {
        ++p;
      }
      if (p >= limit) {
      } else if (*p == 'h' || *p == 'H') {
        // Hollerith
        *p = 'h';
      } else if (*p == '_' && p + 1 < limit && (p[1] == '"' || p[1] == '\'')) {
        // kind-prefixed character literal (e.g., 1_"ABC")
      } else {
        // exponent
        for (; p < limit; ++p) {
          *p = ToLowerCaseLetter(*p);
        }
      }
    } else if (*lastChar == '\'' || *lastChar == '"') {
      if (*p == *lastChar) {
        // Character literal without prefix
      } else if (p[1] == *lastChar) {
        // BOZX-prefixed constant
        for (; p < limit; ++p) {
          *p = ToLowerCaseLetter(*p);
        }
      } else {
        // Literal with kind-param prefix name (e.g., K_"ABC").
        for (; *p != *lastChar; ++p) {
          *p = ToLowerCaseLetter(*p);
        }
      }
    } else {
      for (; p < limit; ++p) {
        *p = ToLowerCaseLetter(*p);
      }
    }
  }
  return *this;
}

bool TokenSequence::HasBlanks(std::size_t firstChar) const {
  std::size_t tokens{SizeInTokens()};
  for (std::size_t j{0}; j < tokens; ++j) {
    if (start_[j] >= firstChar && TokenAt(j).IsBlank()) {
      return true;
    }
  }
  return false;
}

bool TokenSequence::HasRedundantBlanks(std::size_t firstChar) const {
  std::size_t tokens{SizeInTokens()};
  bool lastWasBlank{false};
  for (std::size_t j{0}; j < tokens; ++j) {
    bool isBlank{TokenAt(j).IsBlank()};
    if (isBlank && lastWasBlank && start_[j] >= firstChar) {
      return true;
    }
    lastWasBlank = isBlank;
  }
  return false;
}

TokenSequence &TokenSequence::RemoveBlanks(std::size_t firstChar) {
  std::size_t tokens{SizeInTokens()};
  TokenSequence result;
  for (std::size_t j{0}; j < tokens; ++j) {
    if (!TokenAt(j).IsBlank() || start_[j] < firstChar) {
      result.AppendRange(*this, j);
    }
  }
  swap(result);
  return *this;
}

TokenSequence &TokenSequence::RemoveRedundantBlanks(std::size_t firstChar) {
  std::size_t tokens{SizeInTokens()};
  TokenSequence result;
  bool lastWasBlank{false};
  for (std::size_t j{0}; j < tokens; ++j) {
    bool isBlank{TokenAt(j).IsBlank()};
    if (!isBlank || !lastWasBlank || start_[j] < firstChar) {
      result.AppendRange(*this, j);
    }
    lastWasBlank = isBlank;
  }
  swap(result);
  return *this;
}

TokenSequence &TokenSequence::ClipComment(
    const Prescanner &prescanner, bool skipFirst) {
  std::size_t tokens{SizeInTokens()};
  for (std::size_t j{0}; j < tokens; ++j) {
    CharBlock tok{TokenAt(j)};
    if (std::size_t blanks{tok.CountLeadingBlanks()};
        blanks < tok.size() && tok[blanks] == '!') {
      // Retain active compiler directive sentinels (e.g. "!dir$")
      for (std::size_t k{j + 1}; k < tokens && tok.size() <= blanks + 5; ++k) {
        if (tok.begin() + tok.size() == TokenAt(k).begin()) {
          tok.ExtendToCover(TokenAt(k));
        } else {
          break;
        }
      }
      bool isSentinel{false};
      if (tok.size() > blanks + 5) {
        isSentinel = prescanner.IsCompilerDirectiveSentinel(&tok[blanks + 1])
                         .has_value();
      }
      if (isSentinel) {
      } else if (skipFirst) {
        skipFirst = false;
      } else {
        TokenSequence result;
        if (j > 0) {
          result.AppendRange(*this, 0, j - 1);
        }
        swap(result);
        return *this;
      }
    }
  }
  return *this;
}

void TokenSequence::Emit(CookedSource &cooked) const {
  if (auto n{char_.size()}) {
    cooked.Put(&char_[0], n);
    cooked.PutProvenanceMappings(provenances_);
  }
}

llvm::raw_ostream &TokenSequence::Dump(llvm::raw_ostream &o) const {
  o << "TokenSequence has " << char_.size() << " chars; nextStart_ "
    << nextStart_ << '\n';
  for (std::size_t j{0}; j < start_.size(); ++j) {
    o << '[' << j << "] @ " << start_[j] << " '" << TokenAt(j).ToString()
      << "'\n";
  }
  provenances_.Dump(o << "provenances_:\n");
  return o;
}

Provenance TokenSequence::GetCharProvenance(std::size_t offset) const {
  ProvenanceRange range{provenances_.Map(offset)};
  return range.start();
}

Provenance TokenSequence::GetTokenProvenance(
    std::size_t token, std::size_t offset) const {
  return GetCharProvenance(start_[token] + offset);
}

ProvenanceRange TokenSequence::GetTokenProvenanceRange(
    std::size_t token, std::size_t offset) const {
  ProvenanceRange range{provenances_.Map(start_[token] + offset)};
  return range.Prefix(TokenBytes(token) - offset);
}

ProvenanceRange TokenSequence::GetIntervalProvenanceRange(
    std::size_t token, std::size_t tokens) const {
  if (tokens == 0) {
    return {};
  }
  ProvenanceRange range{provenances_.Map(start_[token])};
  while (--tokens > 0 &&
      range.AnnexIfPredecessor(provenances_.Map(start_[++token]))) {
  }
  return range;
}

ProvenanceRange TokenSequence::GetProvenanceRange() const {
  return GetIntervalProvenanceRange(0, start_.size());
}

const TokenSequence &TokenSequence::CheckBadFortranCharacters(
    Messages &messages, const Prescanner &prescanner,
    bool preprocessingOnly) const {
  std::size_t tokens{SizeInTokens()};
  for (std::size_t j{0}; j < tokens; ++j) {
    CharBlock token{TokenAt(j)};
    char ch{token.FirstNonBlank()};
    if (ch != ' ' && !IsValidFortranTokenCharacter(ch)) {
      if (ch == '!') {
        if (prescanner.IsCompilerDirectiveSentinel(token)) {
          continue;
        } else if (j + 1 < tokens &&
            prescanner.IsCompilerDirectiveSentinel(
                TokenAt(j + 1))) { // !dir$, &c.
          ++j;
          continue;
        } else if (preprocessingOnly) {
          continue;
        }
      } else if (ch == '&' && preprocessingOnly) {
        continue;
      }
      if (ch < ' ' || ch >= '\x7f') {
        messages.Say(GetTokenProvenanceRange(j),
            "bad character (0x%02x) in Fortran token"_err_en_US, ch & 0xff);
      } else {
        messages.Say(GetTokenProvenanceRange(j),
            "bad character ('%c') in Fortran token"_err_en_US, ch);
      }
    }
  }
  return *this;
}

bool TokenSequence::BadlyNestedParentheses() const {
  int nesting{0};
  std::size_t tokens{SizeInTokens()};
  for (std::size_t j{0}; j < tokens; ++j) {
    CharBlock token{TokenAt(j)};
    char ch{token.OnlyNonBlank()};
    if (ch == '(') {
      ++nesting;
    } else if (ch == ')') {
      if (nesting-- == 0) {
        break;
      }
    }
  }
  return nesting != 0;
}

const TokenSequence &TokenSequence::CheckBadParentheses(
    Messages &messages) const {
  if (BadlyNestedParentheses()) {
    // There's an error; diagnose it
    std::size_t tokens{SizeInTokens()};
    std::vector<std::size_t> stack;
    for (std::size_t j{0}; j < tokens; ++j) {
      CharBlock token{TokenAt(j)};
      char ch{token.OnlyNonBlank()};
      if (ch == '(') {
        stack.push_back(j);
      } else if (ch == ')') {
        if (stack.empty()) {
          messages.Say(GetTokenProvenanceRange(j), "Unmatched ')'"_err_en_US);
          return *this;
        }
        stack.pop_back();
      }
    }
    CHECK(!stack.empty());
    messages.Say(
        GetTokenProvenanceRange(stack.back()), "Unmatched '('"_err_en_US);
  }
  return *this;
}
} // namespace Fortran::parser
