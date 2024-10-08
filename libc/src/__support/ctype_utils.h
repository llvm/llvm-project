//===-- Collection of utils for implementing ctype functions-------*-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_CTYPE_UTILS_H
#define LLVM_LIBC_SRC___SUPPORT_CTYPE_UTILS_H

#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {

// -----------------------------------------------------------------------------
// ******************                 WARNING                 ******************
// ****************** DO NOT TRY TO OPTIMIZE THESE FUNCTIONS! ******************
// -----------------------------------------------------------------------------
// This switch/case form is easier for the compiler to understand, and is
// optimized into a form that is almost always the same as or better than
// versions written by hand (see https://godbolt.org/z/qvrebqvvr). Also this
// form makes these functions encoding independent. If you want to rewrite these
// functions, make sure you have benchmarks to show your new solution is faster,
// as well as a way to support non-ASCII character encodings.

LIBC_INLINE static constexpr bool islower(int ch) {
  switch (ch) {
  case 'a':
  case 'b':
  case 'c':
  case 'd':
  case 'e':
  case 'f':
  case 'g':
  case 'h':
  case 'i':
  case 'j':
  case 'k':
  case 'l':
  case 'm':
  case 'n':
  case 'o':
  case 'p':
  case 'q':
  case 'r':
  case 's':
  case 't':
  case 'u':
  case 'v':
  case 'w':
  case 'x':
  case 'y':
  case 'z':
    return true;
  default:
    return false;
  }
}

LIBC_INLINE static constexpr bool isupper(int ch) {
  switch (ch) {
  case 'A':
  case 'B':
  case 'C':
  case 'D':
  case 'E':
  case 'F':
  case 'G':
  case 'H':
  case 'I':
  case 'J':
  case 'K':
  case 'L':
  case 'M':
  case 'N':
  case 'O':
  case 'P':
  case 'Q':
  case 'R':
  case 'S':
  case 'T':
  case 'U':
  case 'V':
  case 'W':
  case 'X':
  case 'Y':
  case 'Z':
    return true;
  default:
    return false;
  }
}

LIBC_INLINE static constexpr bool isdigit(int ch) {
  switch (ch) {
  case '0':
  case '1':
  case '2':
  case '3':
  case '4':
  case '5':
  case '6':
  case '7':
  case '8':
  case '9':
    return true;
  default:
    return false;
  }
}

LIBC_INLINE static constexpr int tolower(int ch) {
  switch (ch) {
  case 'A':
    return 'a';
  case 'B':
    return 'b';
  case 'C':
    return 'c';
  case 'D':
    return 'd';
  case 'E':
    return 'e';
  case 'F':
    return 'f';
  case 'G':
    return 'g';
  case 'H':
    return 'h';
  case 'I':
    return 'i';
  case 'J':
    return 'j';
  case 'K':
    return 'k';
  case 'L':
    return 'l';
  case 'M':
    return 'm';
  case 'N':
    return 'n';
  case 'O':
    return 'o';
  case 'P':
    return 'p';
  case 'Q':
    return 'q';
  case 'R':
    return 'r';
  case 'S':
    return 's';
  case 'T':
    return 't';
  case 'U':
    return 'u';
  case 'V':
    return 'v';
  case 'W':
    return 'w';
  case 'X':
    return 'x';
  case 'Y':
    return 'y';
  case 'Z':
    return 'z';
  default:
    return ch;
  }
}

LIBC_INLINE static constexpr bool isalpha(int ch) {
  switch (tolower(ch)) {
  case 'a':
  case 'b':
  case 'c':
  case 'd':
  case 'e':
  case 'f':
  case 'g':
  case 'h':
  case 'i':
  case 'j':
  case 'k':
  case 'l':
  case 'm':
  case 'n':
  case 'o':
  case 'p':
  case 'q':
  case 'r':
  case 's':
  case 't':
  case 'u':
  case 'v':
  case 'w':
  case 'x':
  case 'y':
  case 'z':
    return true;
  default:
    return false;
  }
}

LIBC_INLINE static constexpr bool isalnum(int ch) {
  return isalpha(ch) || isdigit(ch);
}

LIBC_INLINE static constexpr int b36_char_to_int(int ch) {
  switch (tolower(ch)) {
  case '0':
    return 0;
  case '1':
    return 1;
  case '2':
    return 2;
  case '3':
    return 3;
  case '4':
    return 4;
  case '5':
    return 5;
  case '6':
    return 6;
  case '7':
    return 7;
  case '8':
    return 8;
  case '9':
    return 9;
  case 'a':
    return 10;
  case 'b':
    return 11;
  case 'c':
    return 12;
  case 'd':
    return 13;
  case 'e':
    return 14;
  case 'f':
    return 15;
  case 'g':
    return 16;
  case 'h':
    return 17;
  case 'i':
    return 18;
  case 'j':
    return 19;
  case 'k':
    return 20;
  case 'l':
    return 21;
  case 'm':
    return 22;
  case 'n':
    return 23;
  case 'o':
    return 24;
  case 'p':
    return 25;
  case 'q':
    return 26;
  case 'r':
    return 27;
  case 's':
    return 28;
  case 't':
    return 29;
  case 'u':
    return 30;
  case 'v':
    return 31;
  case 'w':
    return 32;
  case 'x':
    return 33;
  case 'y':
    return 34;
  case 'z':
    return 35;
  default:
    return 0;
  }
}

LIBC_INLINE static constexpr bool isspace(int ch) {
  switch (ch) {
  case ' ':
  case '\t':
  case '\n':
  case '\v':
  case '\f':
  case '\r':
    return true;
  default:
    return false;
  }
}

// not yet encoding independent.
LIBC_INLINE static constexpr bool isgraph(int ch) {
  return 0x20 < ch && ch < 0x7f;
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

#endif //  LLVM_LIBC_SRC___SUPPORT_CTYPE_UTILS_H
