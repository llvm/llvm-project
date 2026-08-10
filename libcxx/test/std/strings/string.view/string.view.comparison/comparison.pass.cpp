//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Starting with C++20 the spaceship operator was included. This tests
// comparison in that context, thus doesn't support older language versions.
// These are tested per operator.

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <string_view>

// template<class charT, class traits>
//   constexpr bool operator==(basic_string_view<charT, traits> lhs, basic_string_view<charT, traits> rhs);
// template<class charT, class traits>
//   constexpr auto operator<=>(basic_string_view<charT, traits> lhs, basic_string_view<charT, traits> rhs);
// (plus "sufficient additional overloads" to make implicit conversions work as intended)

#include <string_view>

#include <array>
#include <cassert>
#include <string>

#include "constexpr_char_traits.h"
#include "make_string.h"
#include "test_comparisons.h"
#include "test_macros.h"

#define SV(S) MAKE_STRING_VIEW(CharT, S)

// Copied from constexpr_char_traits, but it doesn't have a full implementation.
// It has a comparison_category used in the tests.
template <class CharT, class Ordering>
struct char_traits {
  using char_type           = CharT;
  using int_type            = int;
  using off_type            = std::streamoff;
  using pos_type            = std::streampos;
  using state_type          = std::mbstate_t;
  using comparison_category = Ordering;

  static constexpr void assign(char_type& __c1, const char_type& __c2) noexcept { __c1 = __c2; }
  static constexpr bool eq(char_type __c1, char_type __c2) noexcept { return __c1 == __c2; }
  static constexpr bool lt(char_type __c1, char_type __c2) noexcept { return __c1 < __c2; }
  static constexpr int compare(const char_type* __s1, const char_type* __s2, std::size_t __n) {
    for (; __n; --__n, ++__s1, ++__s2) {
      if (lt(*__s1, *__s2))
        return -1;
      if (lt(*__s2, *__s1))
        return 1;
    }
    return 0;
  }

  static constexpr std::size_t length(const char_type* __s);
  static constexpr const char_type* find(const char_type* __s, std::size_t __n, const char_type& __a);
  static constexpr char_type* move(char_type* __s1, const char_type* __s2, std::size_t __n);
  static constexpr char_type* copy(char_type* __s1, const char_type* __s2, std::size_t __n);
  static constexpr char_type* assign(char_type* __s, std::size_t __n, char_type __a);
  static constexpr int_type not_eof(int_type __c) noexcept { return eq_int_type(__c, eof()) ? ~eof() : __c; }
  static constexpr char_type to_char_type(int_type __c) noexcept { return char_type(__c); }
  static constexpr int_type to_int_type(char_type __c) noexcept { return int_type(__c); }
  static constexpr bool eq_int_type(int_type __c1, int_type __c2) noexcept { return __c1 == __c2; }
  static constexpr int_type eof() noexcept { return int_type(EOF); }
};

template <class T, class Ordering = std::strong_ordering>
constexpr void test() {
  AssertOrderAreNoexcept<T>();
  AssertOrderReturn<Ordering, T>();

  using CharT = typename T::value_type;

  // sorted values
  std::array v{
      SV(""),
      SV("abc"),
      SV("abcdef"),
  };

  // sorted values with embedded NUL character
  std::array vn{
      SV("abc"),
      SV("abc\0"),
      SV("abc\0def"),
  };
  static_assert(v.size() == vn.size());

  for (std::size_t i = 0; i < v.size(); ++i) {
    for (std::size_t j = 0; j < v.size(); ++j) {
      assert(testOrder(v[i], v[j], i == j ? Ordering::equivalent : i < j ? Ordering::less : Ordering::greater));
      assert(testOrder(
          v[i],
          std::basic_string<CharT>{v[j]},
          i == j  ? Ordering::equivalent
          : i < j ? Ordering::less
                  : Ordering::greater));

      assert(testOrder(
          v[i],
          std::basic_string<CharT>{v[j]}.c_str(),
          i == j  ? Ordering::equivalent
          : i < j ? Ordering::less
                  : Ordering::greater));

      // NUL test omitted for c-strings since it will fail.
      assert(testOrder(vn[i], vn[j], i == j ? Ordering::equivalent : i < j ? Ordering::less : Ordering::greater));
      assert(testOrder(
          vn[i],
          std::basic_string<CharT>{vn[j]},
          i == j  ? Ordering::equivalent
          : i < j ? Ordering::less
                  : Ordering::greater));
    }
  }
}

template <class CharT>
constexpr void test_all_orderings() {
  test<std::basic_string_view<CharT>>(); // Strong ordering in its char_traits
  test<std::basic_string_view<CharT, constexpr_char_traits<CharT>>,
       std::weak_ordering>(); // No ordering in its char_traits
  test<std::basic_string_view<CharT, char_traits<CharT, std::weak_ordering>>, std::weak_ordering>();
  test<std::basic_string_view<CharT, char_traits<CharT, std::partial_ordering>>, std::partial_ordering>();
}

constexpr bool test_all_types() {
  test_all_orderings<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_all_orderings<wchar_t>();
#endif
  test_all_orderings<char8_t>();
  test_all_orderings<char16_t>();
  test_all_orderings<char32_t>();

  return true;
}

int main(int, char**) {
  test_all_types();
  static_assert(test_all_types());

  return 0;
}
