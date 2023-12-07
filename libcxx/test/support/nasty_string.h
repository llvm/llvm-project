//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_SUPPORT_NASTY_STRING_H
#define TEST_SUPPORT_NASTY_STRING_H

#include <algorithm>
#include <cstddef>
#include <string>
#include <type_traits>

#include "make_string.h"
#include "test_macros.h"

// This defines a nasty_string similar to nasty_containers. This string's
// value_type does operator hijacking, which allows us to ensure that the
// library uses the provided `CharTraits` instead of using operations on
// the value_type directly.


// When using the code during constant evaluation it relies on
//   P2647R1 Permitting static constexpr variables in constexpr functions
// This is a C++23 feature, which is not supported by all compilers yet.
// * GCC >= 13
// * Clang >= 16
// * MSVC no support yet
//
// TODO After there is proper compiler support use TEST_STD_VER >= 23 instead
// of this macro in the tests.
#if TEST_STD_VER < 23 || __cpp_constexpr < 202211L
#  define TEST_HAS_NO_NASTY_STRING
#endif

#ifndef TEST_HAS_NO_NASTY_STRING
// Make sure the char-like operations in strings do not depend on the char-like type.
struct nasty_char {
  template <typename T>
  friend auto operator<=>(T, T) = delete;

  template <typename T>
  friend void operator+(T&&) = delete;

  template <typename T>
  friend void operator-(T&&) = delete;

  template <typename T>
  friend void operator&(T&&) = delete;

  char c;
};

static_assert(std::is_trivial<nasty_char>::value, "");
static_assert(std::is_standard_layout<nasty_char>::value, "");

// These traits are based on the constexpr_traits test class.
struct nasty_char_traits {
  typedef nasty_char char_type;
  typedef int int_type;
  typedef std::streamoff off_type;
  typedef std::streampos pos_type;
  typedef std::mbstate_t state_type;
  // The comparison_category is omitted so the class will have weak_ordering
  // in C++20. This is intentional.

  static constexpr void assign(char_type& c1, const char_type& c2) noexcept { c1 = c2; }

  static constexpr bool eq(char_type c1, char_type c2) noexcept { return c1.c == c2.c; }

  static constexpr bool lt(char_type c1, char_type c2) noexcept { return c1.c < c2.c; }

  static constexpr int compare(const char_type* s1, const char_type* s2, std::size_t n);
  static constexpr std::size_t length(const char_type* s);
  static constexpr const char_type* find(const char_type* s, std::size_t n, const char_type& a);
  static constexpr char_type* move(char_type* s1, const char_type* s2, std::size_t n);
  static constexpr char_type* copy(char_type* s1, const char_type* s2, std::size_t n);
  static constexpr char_type* assign(char_type* s, std::size_t n, char_type a);

  static constexpr int_type not_eof(int_type c) noexcept { return eq_int_type(c, eof()) ? ~eof() : c; }

  static constexpr char_type to_char_type(int_type c) noexcept { return char_type(c); }

  static constexpr int_type to_int_type(char_type c) noexcept { return int_type(c.c); }

  static constexpr bool eq_int_type(int_type c1, int_type c2) noexcept { return c1 == c2; }

  static constexpr int_type eof() noexcept { return int_type(EOF); }
};

constexpr int nasty_char_traits::compare(const nasty_char* s1, const nasty_char* s2, std::size_t n) {
  for (; n; --n, ++s1, ++s2) {
    if (lt(*s1, *s2))
      return -1;
    if (lt(*s2, *s1))
      return 1;
  }
  return 0;
}

constexpr std::size_t nasty_char_traits::length(const nasty_char* s) {
  std::size_t len = 0;
  for (; !eq(*s, nasty_char(0)); ++s)
    ++len;
  return len;
}

constexpr const nasty_char* nasty_char_traits::find(const nasty_char* s, std::size_t n, const nasty_char& a) {
  for (; n; --n) {
    if (eq(*s, a))
      return s;
    ++s;
  }
  return 0;
}

constexpr nasty_char* nasty_char_traits::move(nasty_char* s1, const nasty_char* s2, std::size_t n) {
  nasty_char* r = s1;
  if (s1 < s2) {
    for (; n; --n, ++s1, ++s2)
      assign(*s1, *s2);
  } else if (s2 < s1) {
    s1 += n;
    s2 += n;
    for (; n; --n)
      assign(*--s1, *--s2);
  }
  return r;
}

constexpr nasty_char* nasty_char_traits::copy(nasty_char* s1, const nasty_char* s2, std::size_t n) {
  if (!std::is_constant_evaluated()) // fails in constexpr because we might be comparing unrelated pointers
    assert(s2 < s1 || s2 >= s1 + n);
  nasty_char* r = s1;
  for (; n; --n, ++s1, ++s2)
    assign(*s1, *s2);
  return r;
}

constexpr nasty_char* nasty_char_traits::assign(nasty_char* s, std::size_t n, nasty_char a) {
  nasty_char* r = s;
  for (; n; --n, ++s)
    assign(*s, a);
  return r;
}

using nasty_string = std::basic_string<nasty_char, nasty_char_traits>;

template <std::size_t N>
struct ToNastyChar {
  constexpr ToNastyChar(const char (&r)[N]) {
    std::transform(r, r + N, std::addressof(text[0]), [](char c) { return nasty_char{c}; });
  }
  nasty_char text[N];
};

template <std::size_t N>
ToNastyChar(const char (&)[N]) -> ToNastyChar<N>;

template <ToNastyChar t>
constexpr auto to_nasty_char() {
  return t;
}

// A macro like MAKE_CSTRING
//
// The difference is this macro can convert the nasty_char too.
//
// The lambda is a template, so the 'if constexpr' false branch is not evaluated for the nasty_char.
#  define CONVERT_TO_CSTRING(CHAR, STR)                                                                                \
    []<class CharT> {                                                                                                  \
      if constexpr (std::is_same_v<CharT, nasty_char>) {                                                               \
        static constexpr auto result = to_nasty_char<STR>();                                                           \
        return result.text;                                                                                            \
      } else                                                                                                           \
        return MAKE_CSTRING(CharT, STR);                                                                               \
    }.template operator()<CHAR>() /* */
#else                             // TEST_HAS_NO_NASTY_STRING
#  define CONVERT_TO_CSTRING(CharT, STR) MAKE_CSTRING(CharT, STR)
#endif                            // TEST_HAS_NO_NASTY_STRING

#endif                            // TEST_SUPPORT_NASTY_STRING_H
