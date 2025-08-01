//===-- Collection of utils for implementing wide char functions --*-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_WCTYPE_UTILS_H
#define LLVM_LIBC_SRC___SUPPORT_WCTYPE_UTILS_H

#include "hdr/types/wint_t.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/macros/attributes.h" // LIBC_INLINE
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

// Similarly, do not change these fumarks to show your new solution is faster,
// as well as a way to support non-Anctions to use case ranges. e.g.
//  bool iswlower(wint_t ch) {
//    switch(ch) {
//    case L'a'...L'z':
//      return true;
//    }
//  }
// This assumes the character ranges are contiguous, which they aren't in
// EBCDIC. Technically we could use some smaller ranges, but that's even harder
// to read.

LIBC_INLINE static constexpr bool iswlower(wint_t wch) {
  switch (wch) {
  case L'a':
  case L'b':
  case L'c':
  case L'd':
  case L'e':
  case L'f':
  case L'g':
  case L'h':
  case L'i':
  case L'j':
  case L'k':
  case L'l':
  case L'm':
  case L'n':
  case L'o':
  case L'p':
  case L'q':
  case L'r':
  case L's':
  case L't':
  case L'u':
  case L'v':
  case L'w':
  case L'x':
  case L'y':
  case L'z':
    return true;
  default:
    return false;
  }
}

LIBC_INLINE static constexpr bool iswupper(wint_t wch) {
  switch (wch) {
  case L'A':
  case L'B':
  case L'C':
  case L'D':
  case L'E':
  case L'F':
  case L'G':
  case L'H':
  case L'I':
  case L'J':
  case L'K':
  case L'L':
  case L'M':
  case L'N':
  case L'O':
  case L'P':
  case L'Q':
  case L'R':
  case L'S':
  case L'T':
  case L'U':
  case L'V':
  case L'W':
  case L'X':
  case L'Y':
  case L'Z':
    return true;
  default:
    return false;
  }
}

LIBC_INLINE static constexpr bool iswdigit(wint_t wch) {
  switch (wch) {
  case L'0':
  case L'1':
  case L'2':
  case L'3':
  case L'4':
  case L'5':
  case L'6':
  case L'7':
  case L'8':
  case L'9':
    return true;
  default:
    return false;
  }
}

LIBC_INLINE static constexpr wint_t towlower(wint_t wch) {
  switch (wch) {
  case L'A':
    return L'a';
  case L'B':
    return L'b';
  case L'C':
    return L'c';
  case L'D':
    return L'd';
  case L'E':
    return L'e';
  case L'F':
    return L'f';
  case L'G':
    return L'g';
  case L'H':
    return L'h';
  case L'I':
    return L'i';
  case L'J':
    return L'j';
  case L'K':
    return L'k';
  case L'L':
    return L'l';
  case L'M':
    return L'm';
  case L'N':
    return L'n';
  case L'O':
    return L'o';
  case L'P':
    return L'p';
  case L'Q':
    return L'q';
  case L'R':
    return L'r';
  case L'S':
    return L's';
  case L'T':
    return L't';
  case L'U':
    return L'u';
  case L'V':
    return L'v';
  case L'W':
    return L'w';
  case L'X':
    return L'x';
  case L'Y':
    return L'y';
  case L'Z':
    return L'z';
  default:
    return wch;
  }
}

LIBC_INLINE static constexpr wint_t towupper(wint_t wch) {
  switch (wch) {
  case L'a':
    return L'A';
  case L'b':
    return L'B';
  case L'c':
    return L'C';
  case L'd':
    return L'D';
  case L'e':
    return L'E';
  case L'f':
    return L'F';
  case L'g':
    return L'G';
  case L'h':
    return L'H';
  case L'i':
    return L'I';
  case L'j':
    return L'J';
  case L'k':
    return L'K';
  case L'l':
    return L'L';
  case L'm':
    return L'M';
  case L'n':
    return L'N';
  case L'o':
    return L'O';
  case L'p':
    return L'P';
  case L'q':
    return L'Q';
  case L'r':
    return L'R';
  case L's':
    return L'S';
  case L't':
    return L'T';
  case L'u':
    return L'U';
  case L'v':
    return L'V';
  case L'w':
    return L'W';
  case L'x':
    return L'X';
  case L'y':
    return L'Y';
  case L'z':
    return L'Z';
  default:
    return wch;
  }
}

LIBC_INLINE static constexpr bool iswalpha(wint_t wch) {
  switch (wch) {
  case L'a':
  case L'b':
  case L'c':
  case L'd':
  case L'e':
  case L'f':
  case L'g':
  case L'h':
  case L'i':
  case L'j':
  case L'k':
  case L'l':
  case L'm':
  case L'n':
  case L'o':
  case L'p':
  case L'q':
  case L'r':
  case L's':
  case L't':
  case L'u':
  case L'v':
  case L'w':
  case L'x':
  case L'y':
  case L'z':
  case L'A':
  case L'B':
  case L'C':
  case L'D':
  case L'E':
  case L'F':
  case L'G':
  case L'H':
  case L'I':
  case L'J':
  case L'K':
  case L'L':
  case L'M':
  case L'N':
  case L'O':
  case L'P':
  case L'Q':
  case L'R':
  case L'S':
  case L'T':
  case L'U':
  case L'V':
  case L'W':
  case L'X':
  case L'Y':
  case L'Z':
    return true;
  default:
    return false;
  }
}

LIBC_INLINE static constexpr bool iswalnum(wint_t wch) {
  switch (wch) {
  case L'a':
  case L'b':
  case L'c':
  case L'd':
  case L'e':
  case L'f':
  case L'g':
  case L'h':
  case L'i':
  case L'j':
  case L'k':
  case L'l':
  case L'm':
  case L'n':
  case L'o':
  case L'p':
  case L'q':
  case L'r':
  case L's':
  case L't':
  case L'u':
  case L'v':
  case L'w':
  case L'x':
  case L'y':
  case L'z':
  case L'A':
  case L'B':
  case L'C':
  case L'D':
  case L'E':
  case L'F':
  case L'G':
  case L'H':
  case L'I':
  case L'J':
  case L'K':
  case L'L':
  case L'M':
  case L'N':
  case L'O':
  case L'P':
  case L'Q':
  case L'R':
  case L'S':
  case L'T':
  case L'U':
  case L'V':
  case L'W':
  case L'X':
  case L'Y':
  case L'Z':
  case L'0':
  case L'1':
  case L'2':
  case L'3':
  case L'4':
  case L'5':
  case L'6':
  case L'7':
  case L'8':
  case L'9':
    return true;
  default:
    return false;
  }
}

LIBC_INLINE static constexpr int b36_wchar_to_int(wint_t wch) {
  switch (wch) {
  case L'0':
    return 0;
  case L'1':
    return 1;
  case L'2':
    return 2;
  case L'3':
    return 3;
  case L'4':
    return 4;
  case L'5':
    return 5;
  case L'6':
    return 6;
  case L'7':
    return 7;
  case L'8':
    return 8;
  case L'9':
    return 9;
  case L'a':
  case L'A':
    return 10;
  case L'b':
  case L'B':
    return 11;
  case L'c':
  case L'C':
    return 12;
  case L'd':
  case L'D':
    return 13;
  case L'e':
  case L'E':
    return 14;
  case L'f':
  case L'F':
    return 15;
  case L'g':
  case L'G':
    return 16;
  case L'h':
  case L'H':
    return 17;
  case L'i':
  case L'I':
    return 18;
  case L'j':
  case L'J':
    return 19;
  case L'k':
  case L'K':
    return 20;
  case L'l':
  case L'L':
    return 21;
  case L'm':
  case L'M':
    return 22;
  case L'n':
  case L'N':
    return 23;
  case L'o':
  case L'O':
    return 24;
  case L'p':
  case L'P':
    return 25;
  case L'q':
  case L'Q':
    return 26;
  case L'r':
  case L'R':
    return 27;
  case L's':
  case L'S':
    return 28;
  case L't':
  case L'T':
    return 29;
  case L'u':
  case L'U':
    return 30;
  case L'v':
  case L'V':
    return 31;
  case L'w':
  case L'W':
    return 32;
  case L'x':
  case L'X':
    return 33;
  case L'y':
  case L'Y':
    return 34;
  case L'z':
  case L'Z':
    return 35;
  default:
    return 0;
  }
}

LIBC_INLINE static constexpr wint_t int_to_b36_wchar(int num) {
  // Can't actually use LIBC_ASSERT here because it depends on integer_to_string
  // which depends on this.

  // LIBC_ASSERT(num < 36);
  switch (num) {
  case 0:
    return L'0';
  case 1:
    return L'1';
  case 2:
    return L'2';
  case 3:
    return L'3';
  case 4:
    return L'4';
  case 5:
    return L'5';
  case 6:
    return L'6';
  case 7:
    return L'7';
  case 8:
    return L'8';
  case 9:
    return L'9';
  case 10:
    return L'a';
  case 11:
    return L'b';
  case 12:
    return L'c';
  case 13:
    return L'd';
  case 14:
    return L'e';
  case 15:
    return L'f';
  case 16:
    return L'g';
  case 17:
    return L'h';
  case 18:
    return L'i';
  case 19:
    return L'j';
  case 20:
    return L'k';
  case 21:
    return L'l';
  case 22:
    return L'm';
  case 23:
    return L'n';
  case 24:
    return L'o';
  case 25:
    return L'p';
  case 26:
    return L'q';
  case 27:
    return L'r';
  case 28:
    return L's';
  case 29:
    return L't';
  case 30:
    return L'u';
  case 31:
    return L'v';
  case 32:
    return L'w';
  case 33:
    return L'x';
  case 34:
    return L'y';
  case 35:
    return L'z';
  default:
    return L'!';
  }
}

LIBC_INLINE static constexpr bool iswspace(wint_t wch) {
  switch (wch) {
  case L' ':
  case L'\t':
  case L'\n':
  case L'\v':
  case L'\f':
  case L'\r':
    return true;
  default:
    return false;
  }
}

// ------------------------------------------------------
// Rationale: Since these classification functions are
// called in other functions, we will avoid the overhead
// of a function call by inlining them.
// ------------------------------------------------------

LIBC_INLINE cpp::optional<int> wctob(wint_t c) {
  // This needs to be translated to EOF at the callsite. This is to avoid
  // including stdio.h in this file.
  // The standard states that wint_t may either be an alias of wchar_t or
  // an alias of an integer type, different platforms define this type with
  // different signedness. This is equivalent to `(c > 127) || (c < 0)` but also
  // works without -Wtype-limits warnings when `wint_t` is unsigned.
  if ((c & ~127) != 0)
    return cpp::nullopt;
  return static_cast<int>(c);
}

LIBC_INLINE cpp::optional<wint_t> btowc(int c) {
  if (c > 127 || c < 0)
    return cpp::nullopt;
  return static_cast<wint_t>(c);
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_WCTYPE_UTILS_H
