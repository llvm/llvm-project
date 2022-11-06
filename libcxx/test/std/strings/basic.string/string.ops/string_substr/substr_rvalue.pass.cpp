//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <string>

// constexpr basic_string substr(size_type pos = 0, size_type n = npos) &&;

#include <algorithm>
#include <string>

#include "constexpr_char_traits.h"
#include "make_string.h"
#include "min_allocator.h"
#include "test_allocator.h"

#define STR(string) MAKE_CSTRING(typename S::value_type, string)

constexpr struct should_throw_exception_t {
} should_throw_exception;

template <class S>
constexpr void test(S orig, size_t pos, ptrdiff_t n, S expected) {
  S str = std::move(orig).substr(pos, n);
  LIBCPP_ASSERT(orig.__invariants());
  LIBCPP_ASSERT(str.__invariants());
  assert(str == expected);
}

template <class S>
constexpr void test(S orig, size_t pos, ptrdiff_t n, should_throw_exception_t) {
#ifndef TEST_HAS_NO_EXCEPTIONS
  if (!std::is_constant_evaluated()) {
    try {
      S str = std::move(orig).substr(pos, n);
      assert(false);
    } catch (const std::out_of_range&) {
    }
  }
#else
  (void)orig;
  (void)pos;
  (void)n;
#endif
}

template <class S>
constexpr void test_string() {
  test<S>(STR(""), 0, 0, STR(""));
  test<S>(STR(""), 0, 1, STR(""));
  test<S>(STR(""), 1, 0, should_throw_exception);
  test<S>(STR(""), 1, 1, should_throw_exception);
  test<S>(STR("short string"), 0, 1, STR("s"));
  test<S>(STR("short string"), 5, 5, STR(" stri"));
  test<S>(STR("short string"), 12, 5, STR(""));
  test<S>(STR("short string"), 13, 5, should_throw_exception);
  test<S>(STR("long long string so no SSO"), 0, 0, STR(""));
  test<S>(STR("long long string so no SSO"), 0, 10, STR("long long "));
  test<S>(STR("long long string so no SSO"), 10, 10, STR("string so "));
  test<S>(STR("long long string so no SSO"), 20, 10, STR("no SSO"));
  test<S>(STR("long long string so no SSO"), 26, 10, STR(""));
  test<S>(STR("long long string so no SSO"), 27, 0, should_throw_exception);
}

template <class CharT, class CharTraits>
constexpr void test_allocators() {
  test_string<std::basic_string<CharT, CharTraits, std::allocator<CharT>>>();
  test_string<std::basic_string<CharT, CharTraits, min_allocator<CharT>>>();
  test_string<std::basic_string<CharT, CharTraits, test_allocator<CharT>>>();
}

template <class CharT>
constexpr void test_char_traits() {
  test_allocators<CharT, std::char_traits<CharT>>();
  test_allocators<CharT, constexpr_char_traits<CharT>>();
}

constexpr bool test() {
  test_char_traits<char>();
  test_char_traits<char16_t>();
  test_char_traits<char32_t>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_char_traits<wchar_t>();
#endif
#ifndef TEST_HAS_NO_CHAR8_T
  test_char_traits<char8_t>();
#endif

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
