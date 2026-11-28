//===-- Collection of utils for implementing wide char functions --*-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_WCTYPE_UTILS_H
#define LLVM_LIBC_SRC___SUPPORT_WCTYPE_UTILS_H

#include "hdr/types/wchar_t.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/macros/attributes.h" // LIBC_INLINE
#include "src/__support/macros/config.h"
#include "src/__support/wctype_basics.h"

#define LIBC_WCTYPE_MODE_ASCII 0
#define LIBC_WCTYPE_MODE_UTF8 1

#ifndef LIBC_CONF_WCTYPE_MODE
#define LIBC_CONF_WCTYPE_MODE LIBC_WCTYPE_MODE_ASCII
#endif

#if LIBC_CONF_WCTYPE_MODE == LIBC_WCTYPE_MODE_UTF8
#include "src/__support/wctype/wctype_classification_utils.h"
#include "src/__support/wctype/wctype_conversion_utils.h"
#endif

namespace LIBC_NAMESPACE_DECL {
namespace internal {

LIBC_INLINE constexpr bool islower(wchar_t wch) {
#if LIBC_CONF_WCTYPE_MODE != LIBC_WCTYPE_MODE_UTF8
  return ascii::islower(wch);
#else
  if (static_cast<uint32_t>(wch) < 128) {
    return ascii::islower(wch);
  }
  return lookup_properties(wch) & PropertyFlag::LOWER;
#endif
}

LIBC_INLINE constexpr bool isupper(wchar_t wch) {
#if LIBC_CONF_WCTYPE_MODE != LIBC_WCTYPE_MODE_UTF8
  return ascii::isupper(wch);
#else
  if (static_cast<uint32_t>(wch) < 128) {
    return ascii::isupper(wch);
  }
  return lookup_properties(wch) & PropertyFlag::UPPER;
#endif
}

LIBC_INLINE constexpr bool isdigit(wchar_t wch) {
  // In C.UT8, only ASCII digits are considered digits
  return ascii::isdigit(wch);
}

LIBC_INLINE constexpr bool isalpha(wchar_t wch) {
#if LIBC_CONF_WCTYPE_MODE != LIBC_WCTYPE_MODE_UTF8
  return ascii::isalpha(wch);
#else
  if (static_cast<uint32_t>(wch) < 128) {
    return ascii::isalpha(wch);
  }
  return lookup_properties(wch) & PropertyFlag::ALPHA;
#endif
}

LIBC_INLINE constexpr bool isalnum(wchar_t wch) {
#if LIBC_CONF_WCTYPE_MODE != LIBC_WCTYPE_MODE_UTF8
  return ascii::isalnum(wch);
#else
  if (static_cast<uint32_t>(wch) < 128) {
    return ascii::isalnum(wch);
  }
  // Only need to check ALPHA, digit cases are covered by ASCII path
  return lookup_properties(wch) & PropertyFlag::ALPHA;
#endif
}

LIBC_INLINE constexpr bool isspace(wchar_t wch) {
#if LIBC_CONF_WCTYPE_MODE != LIBC_WCTYPE_MODE_UTF8
  return ascii::isspace(wch);
#else
  if (static_cast<uint32_t>(wch) < 128) {
    return ascii::isspace(wch);
  }
  return lookup_properties(wch) & PropertyFlag::SPACE;
#endif
}

LIBC_INLINE constexpr bool isblank(wchar_t wch) {
#if LIBC_CONF_WCTYPE_MODE != LIBC_WCTYPE_MODE_UTF8
  return ascii::isblank(wch);
#else
  if (static_cast<uint32_t>(wch) < 128) {
    return ascii::isblank(wch);
  }
  return lookup_properties(wch) & PropertyFlag::BLANK;
#endif
}

LIBC_INLINE constexpr bool isgraph(wchar_t wch) {
#if LIBC_CONF_WCTYPE_MODE != LIBC_WCTYPE_MODE_UTF8
  return ascii::isgraph(wch);
#else
  if (static_cast<uint32_t>(wch) < 128) {
    return ascii::isgraph(wch);
  }
  // print && !space
  return (lookup_properties(wch) &
          (PropertyFlag::PRINT | PropertyFlag::SPACE)) == PropertyFlag::PRINT;
#endif
}

LIBC_INLINE constexpr bool isprint(wchar_t wch) {
#if LIBC_CONF_WCTYPE_MODE != LIBC_WCTYPE_MODE_UTF8
  return ascii::isprint(wch);
#else
  if (static_cast<uint32_t>(wch) < 128) {
    return ascii::isprint(wch);
  }
  return lookup_properties(wch) & PropertyFlag::PRINT;
#endif
}
LIBC_INLINE constexpr bool isxdigit(wchar_t wch) {
  // Hexadecimal digits are the same in C.UTF8 as in ASCII
  return ascii::isxdigit(wch);
}

LIBC_INLINE constexpr bool iscntrl(wchar_t wch) {
#if LIBC_CONF_WCTYPE_MODE != LIBC_WCTYPE_MODE_UTF8
  return ascii::iscntrl(wch);
#else
  if (static_cast<uint32_t>(wch) < 128) {
    return ascii::iscntrl(wch);
  }
  return lookup_properties(wch) & PropertyFlag::CNTRL;
#endif
}

LIBC_INLINE constexpr bool ispunct(wchar_t wch) {
#if LIBC_CONF_WCTYPE_MODE != LIBC_WCTYPE_MODE_UTF8
  return ascii::ispunct(wch);
#else
  if (static_cast<uint32_t>(wch) < 128) {
    return ascii::ispunct(wch);
  }
  return lookup_properties(wch) & PropertyFlag::PUNCT;
#endif
}

LIBC_INLINE constexpr wchar_t tolower(wchar_t wch) {
#if LIBC_CONF_WCTYPE_MODE != LIBC_WCTYPE_MODE_UTF8
  return ascii::tolower(wch);
#else
  if (static_cast<uint32_t>(wch) < 128) {
    return ascii::tolower(wch);
  }
  return static_cast<wchar_t>(wctype_internal::tolower(wch));
#endif
}

LIBC_INLINE constexpr wchar_t toupper(wchar_t wch) {
#if LIBC_CONF_WCTYPE_MODE != LIBC_WCTYPE_MODE_UTF8
  return ascii::toupper(wch);
#else
  if (static_cast<uint32_t>(wch) < 128) {
    return ascii::toupper(wch);
  }
  return static_cast<wchar_t>(wctype_internal::toupper(wch));
#endif
}

LIBC_INLINE constexpr int b36_char_to_int(wchar_t wch) {
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

// An overload which provides a way to compare input with specific character
// values, when input can be of a regular or a wide character type.
LIBC_INLINE static constexpr bool
is_char_or_wchar(wchar_t ch, [[maybe_unused]] char, wchar_t wc_value) {
  return (ch == wc_value);
}

// Returns a three-way comparison between two wide character code points.
LIBC_INLINE int threeway_cmp_single(wchar_t left, wchar_t right) {
  // Valid UTF-32 code points are in [0, 0x10FFFF]. If invalid code points were
  // treated as UB, the comparison below could done with 32-bit arithmetic.
  if (left == right)
    return 0;
  if (left < right)
    return -1;
  return 1;
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_WCTYPE_UTILS_H
