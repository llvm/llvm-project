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

// <string>

// template<class charT, class traits, class Allocator>
//   see below operator<=>(const basic_string<charT, traits, Allocator>& lhs,
//                         const basic_string<charT, traits, Allocator>& rhs) noexcept;
// template<class charT, class traits, class Allocator>
//   see below operator<=>(const basic_string<charT, traits, Allocator>& lhs,
//                         const charT* rhs);

#include <string>

#include <array>
#include <cassert>
#include <string_view>

#include "constexpr_char_traits.h"
#include "make_string.h"
#include "test_comparisons.h"
#include "test_macros.h"

#define STR(S) MAKE_STRING(CharT, S)

template <class T, class Ordering = std::strong_ordering>
constexpr void test() {
  AssertOrderAreNoexcept<T>();
  AssertOrderReturn<Ordering, T>();

  using CharT = typename T::value_type;
  AssertOrderReturn<Ordering, T, const CharT*>();

  // sorted values
  std::array v{
      STR(""),
      STR("abc"),
      STR("abcdef"),
      STR("acb"),
  };

  // sorted values with embedded NUL character
  std::array vn{
      STR("abc"),
      STR("abc\0"),
      STR("abc\0def"),
      STR("acb\0"),
  };
  static_assert(v.size() == vn.size());

  for (size_t i = 0; i < v.size(); ++i) {
    for (size_t j = 0; j < v.size(); ++j) {
      assert(testOrder(v[i], v[j], i == j ? Ordering::equivalent : i < j ? Ordering::less : Ordering::greater));

      assert(testOrder(
          v[i],
          std::basic_string<CharT>{v[j]}.c_str(),
          i == j  ? Ordering::equivalent
          : i < j ? Ordering::less
                  : Ordering::greater));

      // NUL test omitted for c-strings since it will fail.
      assert(testOrder(vn[i], vn[j], i == j ? Ordering::equivalent : i < j ? Ordering::less : Ordering::greater));
    }
  }
}

constexpr bool test_all_types() {
  test<std::string>();
  test<std::basic_string<char, constexpr_char_traits<char>>, std::weak_ordering>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<std::wstring>();
  test<std::basic_string<wchar_t, constexpr_char_traits<wchar_t>>, std::weak_ordering>();
#endif
  test<std::u8string>();
  test<std::basic_string<char8_t, constexpr_char_traits<char8_t>>, std::weak_ordering>();
  test<std::u16string>();
  test<std::basic_string<char16_t, constexpr_char_traits<char16_t>>, std::weak_ordering>();
  test<std::u32string>();
  test<std::basic_string<char32_t, constexpr_char_traits<char32_t>>, std::weak_ordering>();

  return true;
}

int main(int, char**) {
  test_all_types();
  static_assert(test_all_types());

  return 0;
}
