//===-- include/flang/Parser/char-block.h -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_PARSER_CHAR_BLOCK_H_
#define FORTRAN_PARSER_CHAR_BLOCK_H_

// Describes a contiguous block of characters; does not own their storage.

#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <string>

namespace llvm {
class raw_ostream;
}

namespace Fortran::parser {

class CharBlock : public llvm::StringRef {
public:
  using llvm::StringRef::StringRef;
  CharBlock(const char *begin, const char *end)
      : llvm::StringRef(begin, end - begin) {}
  CharBlock(const char *begin) : llvm::StringRef(begin, 1) {}

  bool Contains(const CharBlock &that) const {
    if (empty() || that.empty()) {
      return !empty() || that.empty();
    }
    return begin() <= that.begin() && that.end() <= end();
  }

  void ExtendToCover(const CharBlock &that) {
    if (empty()) {
      *this = that;
    } else if (!that.empty()) {
      *this = CharBlock(
          std::min(begin(), that.begin()), std::max(end(), that.end()));
    }
  }

  // Returns the block's first non-blank character, if it has
  // one; otherwise ' '.
  char FirstNonBlank() const {
    size_t idx{LocateFirstNonBlank()};
    return idx != npos ? data()[idx] : ' ';
  }

  // Returns the block's only non-blank character, if it has
  // exactly one non-blank character; otherwise ' '.
  char OnlyNonBlank() const {
    char result{' '};
    for (char ch : *this) {
      if (ch != ' ' && ch != '\t') {
        if (result == ' ') {
          result = ch;
        } else {
          return ' ';
        }
      }
    }
    return result;
  }

  size_t CountLeadingBlanks() const {
    size_t idx{LocateFirstNonBlank()};
    return idx != npos ? idx : size();
  }

  bool IsBlank() const { return LocateFirstNonBlank() == npos; }

  std::string ToString() const { return str(); }

  // Convert to string, stopping early at any embedded '\0'.
  std::string NULTerminatedToString() const {
    return std::string{begin(), strnlen(begin(), size())};
  }

private:
  size_t LocateFirstNonBlank() const {
    return find_if_not([](char c) { return c == ' ' || c == '\t'; });
  }
};

// An alternative comparator based on pointer values; use with care!
struct CharBlockPointerComparator {
  bool operator()(CharBlock x, CharBlock y) const {
    return x.end() < y.begin();
  }
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const CharBlock &x);

} // namespace Fortran::parser

// Specializations to enable std::unordered_map<CharBlock, ...> &c.
template <> struct std::hash<Fortran::parser::CharBlock> {
  std::size_t operator()(const Fortran::parser::CharBlock &x) const {
    std::size_t hash{0}, bytes{x.size()};
    for (std::size_t j{0}; j < bytes; ++j) {
      hash = (hash * 31) ^ x[j];
    }
    return hash;
  }
};
#endif // FORTRAN_PARSER_CHAR_BLOCK_H_
