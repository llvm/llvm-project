//===-- include/flang/Parser/char-set.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_PARSER_CHAR_SET_H_
#define FORTRAN_PARSER_CHAR_SET_H_

// Sets of distinct characters that are valid in Fortran programs outside
// character literals are encoded as 64-bit integers by mapping them to a 6-bit
// character set encoding in which the case of letters is lost (even if
// mixed case input reached the parser, which it does not).  These sets
// need to be suitable for constexprs, so std::bitset<> was not eligible.

#include <cinttypes>
#include <string>

namespace Fortran::parser {

struct SetOfChars {
  constexpr SetOfChars() {}

  constexpr SetOfChars(char c) {
    // This is basically the old DECSIX encoding, which maps the
    // 7-bit ASCII codes [32..95] to [0..63].  Only '#', '&', '?', '\', and '^'
    // in that range are unused in Fortran after preprocessing outside
    // character literals.  We repurpose '^' and '?' for newline and unknown
    // characters (resp.), leaving the others alone in case this code might
    // be useful in preprocssing.
    if (c == '\n') {
      // map newline to '^'
      c = '^';
    } else if (c < 32 || c >= 127) {
      // map other control characters, DEL, and 8-bit characters to '?'
      c = '?';
    } else if (c >= 96) {
      // map lower-case letters to upper-case
      c -= 32;
    }
    // range is now [32..95]; reduce to [0..63] and use as a shift count
    bits_ = static_cast<std::uint64_t>(1) << (c - 32);
  }

  constexpr SetOfChars(const char str[], std::size_t n) {
    for (std::size_t j{0}; j < n; ++j) {
      bits_ |= SetOfChars{str[j]}.bits_;
    }
  }

  constexpr SetOfChars(const SetOfChars &) = default;
  constexpr SetOfChars(SetOfChars &&) = default;
  constexpr SetOfChars &operator=(const SetOfChars &) = default;
  constexpr SetOfChars &operator=(SetOfChars &&) = default;
  constexpr bool empty() const { return bits_ == 0; }

  constexpr bool Has(SetOfChars that) const {
    return (that.bits_ & ~bits_) == 0;
  }
  constexpr SetOfChars Union(SetOfChars that) const {
    return SetOfChars{bits_ | that.bits_};
  }
  constexpr SetOfChars Intersection(SetOfChars that) const {
    return SetOfChars{bits_ & that.bits_};
  }
  constexpr SetOfChars Difference(SetOfChars that) const {
    return SetOfChars{bits_ & ~that.bits_};
  }

  std::string ToString() const;

private:
  constexpr SetOfChars(std::uint64_t b) : bits_{b} {}
  std::uint64_t bits_{0};
};
} // namespace Fortran::parser
#endif // FORTRAN_PARSER_CHAR_SET_H_
