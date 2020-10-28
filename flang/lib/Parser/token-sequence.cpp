//===-- lib/Parser/token-sequence.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "token-sequence.h"
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

void TokenSequence::RemoveLastToken() {
  CHECK(!start_.empty());
  CHECK(nextStart_ > start_.back());
  std::size_t bytes{nextStart_ - start_.back()};
  nextStart_ = start_.back();
  start_.pop_back();
  char_.erase(char_.begin() + nextStart_, char_.end());
  provenances_.RemoveLastBytes(bytes);
}

void TokenSequence::Put(const TokenSequence &that) {
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

void TokenSequence::Put(const TokenSequence &that, ProvenanceRange range) {
  std::size_t offset{0};
  std::size_t tokens{that.SizeInTokens()};
  for (std::size_t j{0}; j < tokens; ++j) {
    CharBlock tok{that.TokenAt(j)};
    Put(tok, range.OffsetMember(offset));
    offset += tok.size();
  }
  CHECK(offset == range.size());
}

void TokenSequence::Put(
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
  Put(&t[0], t.size(), provenance);
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
    j = nextStart;
    if (IsDecimalDigit(*p)) {
      while (p < limit && IsDecimalDigit(*p)) {
        ++p;
      }
      if (p >= limit) {
      } else if (*p == 'h' || *p == 'H') {
        // Hollerith
        *p = 'h';
      } else if (*p == '_') {
        // kind-prefixed character literal (e.g., 1_"ABC")
      } else {
        // exponent
        for (; p < limit; ++p) {
          *p = ToLowerCaseLetter(*p);
        }
      }
    } else if (limit[-1] == '\'' || limit[-1] == '"') {
      if (*p == limit[-1]) {
        // Character literal without prefix
      } else if (p[1] == limit[-1]) {
        // BOZX-prefixed constant
        for (; p < limit; ++p) {
          *p = ToLowerCaseLetter(*p);
        }
      } else {
        // Literal with kind-param prefix name (e.g., K_"ABC").
        for (; *p != limit[-1]; ++p) {
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
      result.Put(*this, j);
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
      result.Put(*this, j);
    }
    lastWasBlank = isBlank;
  }
  swap(result);
  return *this;
}

TokenSequence &TokenSequence::ClipComment(bool skipFirst) {
  std::size_t tokens{SizeInTokens()};
  for (std::size_t j{0}; j < tokens; ++j) {
    if (TokenAt(j).FirstNonBlank() == '!') {
      if (skipFirst) {
        skipFirst = false;
      } else {
        TokenSequence result;
        if (j > 0) {
          result.Put(*this, 0, j - 1);
        }
        swap(result);
        return *this;
      }
    }
  }
  return *this;
}

void TokenSequence::Emit(CookedSource &cooked) const {
  cooked.Put(&char_[0], char_.size());
  cooked.PutProvenanceMappings(provenances_);
}

void TokenSequence::Dump(llvm::raw_ostream &o) const {
  o << "TokenSequence has " << char_.size() << " chars; nextStart_ "
    << nextStart_ << '\n';
  for (std::size_t j{0}; j < start_.size(); ++j) {
    o << '[' << j << "] @ " << start_[j] << " '" << TokenAt(j).ToString()
      << "'\n";
  }
}

Provenance TokenSequence::GetTokenProvenance(
    std::size_t token, std::size_t offset) const {
  ProvenanceRange range{provenances_.Map(start_[token] + offset)};
  return range.start();
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
    Messages &messages) const {
  std::size_t tokens{SizeInTokens()};
  for (std::size_t j{0}; j < tokens; ++j) {
    CharBlock token{TokenAt(j)};
    char ch{token.FirstNonBlank()};
    if (ch != ' ' && !IsValidFortranTokenCharacter(ch)) {
      if (ch == '!' && j == 0) {
        // allow in !dir$
      } else if (ch < ' ' || ch >= '\x7f') {
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
} // namespace Fortran::parser
