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

#include "flang/Common/interval.h"
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <iosfwd>
#include <string>
#include <utility>

namespace llvm {
class raw_ostream;
}

namespace Fortran::parser {

class CharBlock {
public:
  constexpr CharBlock() {}
  constexpr CharBlock(const char *x, std::size_t n = 1) : interval_{x, n} {}
  constexpr CharBlock(const char *b, const char *ep1)
      : interval_{b, static_cast<std::size_t>(ep1 - b)} {}
  CharBlock(const std::string &s) : interval_{s.data(), s.size()} {}
  constexpr CharBlock(const CharBlock &) = default;
  constexpr CharBlock(CharBlock &&) = default;
  constexpr CharBlock &operator=(const CharBlock &) = default;
  constexpr CharBlock &operator=(CharBlock &&) = default;

  constexpr bool empty() const { return interval_.empty(); }
  constexpr std::size_t size() const { return interval_.size(); }
  constexpr const char *begin() const { return interval_.start(); }
  constexpr const char *end() const {
    return interval_.start() + interval_.size();
  }
  constexpr const char &operator[](std::size_t j) const {
    return interval_.start()[j];
  }

  bool Contains(const CharBlock &that) const {
    return interval_.Contains(that.interval_);
  }

  void ExtendToCover(const CharBlock &that) {
    interval_.ExtendToCover(that.interval_);
  }

  // Returns the block's first non-blank character, if it has
  // one; otherwise ' '.
  char FirstNonBlank() const {
    for (char ch : *this) {
      if (ch != ' ' && ch != '\t') {
        return ch;
      }
    }
    return ' '; // non no-blank character
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

  std::size_t CountLeadingBlanks() const {
    std::size_t n{size()};
    std::size_t j{0};
    for (; j < n; ++j) {
      char ch{(*this)[j]};
      if (ch != ' ' && ch != '\t') {
        break;
      }
    }
    return j;
  }

  bool IsBlank() const { return FirstNonBlank() == ' '; }

  std::string ToString() const {
    return std::string{interval_.start(), interval_.size()};
  }

  // Convert to string, stopping early at any embedded '\0'.
  std::string NULTerminatedToString() const {
    return std::string{interval_.start(),
        /*not in std::*/ strnlen(interval_.start(), interval_.size())};
  }

  bool operator<(const CharBlock &that) const { return Compare(that) < 0; }
  bool operator<=(const CharBlock &that) const { return Compare(that) <= 0; }
  bool operator==(const CharBlock &that) const { return Compare(that) == 0; }
  bool operator!=(const CharBlock &that) const { return Compare(that) != 0; }
  bool operator>=(const CharBlock &that) const { return Compare(that) >= 0; }
  bool operator>(const CharBlock &that) const { return Compare(that) > 0; }

  bool operator<(const char *that) const { return Compare(that) < 0; }
  bool operator<=(const char *that) const { return Compare(that) <= 0; }
  bool operator==(const char *that) const { return Compare(that) == 0; }
  bool operator!=(const char *that) const { return Compare(that) != 0; }
  bool operator>=(const char *that) const { return Compare(that) >= 0; }
  bool operator>(const char *that) const { return Compare(that) > 0; }

  friend bool operator<(const char *, const CharBlock &);
  friend bool operator<=(const char *, const CharBlock &);
  friend bool operator==(const char *, const CharBlock &);
  friend bool operator!=(const char *, const CharBlock &);
  friend bool operator>=(const char *, const CharBlock &);
  friend bool operator>(const char *, const CharBlock &);

private:
  int Compare(const CharBlock &that) const {
    std::size_t bytes{std::min(size(), that.size())};
    int cmp{std::memcmp(static_cast<const void *>(begin()),
        static_cast<const void *>(that.begin()), bytes)};
    if (cmp != 0) {
      return cmp;
    }
    return size() < that.size() ? -1 : size() > that.size();
  }

  int Compare(const char *that) const {
    std::size_t bytes{size()};
    if (int cmp{std::strncmp(begin(), that, bytes)}) {
      return cmp;
    }
    return that[bytes] == '\0' ? 0 : -1;
  }

  common::Interval<const char *> interval_{nullptr, 0};
};

inline bool operator<(const char *left, const CharBlock &right) {
  return right > left;
}
inline bool operator<=(const char *left, const CharBlock &right) {
  return right >= left;
}
inline bool operator==(const char *left, const CharBlock &right) {
  return right == left;
}
inline bool operator!=(const char *left, const CharBlock &right) {
  return right != left;
}
inline bool operator>=(const char *left, const CharBlock &right) {
  return right <= left;
}
inline bool operator>(const char *left, const CharBlock &right) {
  return right < left;
}

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
