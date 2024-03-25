//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <string>

// constexpr basic_string(basic_string&& str, size_type pos, const Allocator& a = Allocator());
// constexpr basic_string(basic_string&& str, size_type pos, size_type n, const Allocator& a = Allocator());

#include <cassert>
#include <stdexcept>
#include <string>

#include "constexpr_char_traits.h"
#include "count_new.h"
#include "make_string.h"
#include "min_allocator.h"
#include "test_allocator.h"
#include "test_macros.h"
#include "asan_testing.h"

#define STR(string) MAKE_CSTRING(typename S::value_type, string)

constexpr struct should_throw_exception_t {
} should_throw_exception;

template <class S>
constexpr void test_string_pos(S orig, typename S::size_type pos, S expected) {
#ifdef _LIBCPP_VERSION
  ConstexprDisableAllocationGuard g;
#endif
  S substr(std::move(orig), pos);
  LIBCPP_ASSERT(orig.__invariants());
  LIBCPP_ASSERT(orig.empty());
  LIBCPP_ASSERT(substr.__invariants());
  assert(substr == expected);
  LIBCPP_ASSERT(is_string_asan_correct(orig));
  LIBCPP_ASSERT(is_string_asan_correct(substr));
}

template <class S>
constexpr void test_string_pos(S orig, typename S::size_type pos, should_throw_exception_t) {
#ifndef TEST_HAS_NO_EXCEPTIONS
  if (!std::is_constant_evaluated()) {
    try {
      [[maybe_unused]] S substr = S(std::move(orig), pos);
      assert(false);
    } catch (const std::out_of_range&) {
    }
  }
#else
  (void)orig;
  (void)pos;
#endif
}

template <class S>
constexpr void
test_string_pos_alloc(S orig, typename S::size_type pos, const typename S::allocator_type& alloc, S expected) {
  S substr(std::move(orig), pos, alloc);
  LIBCPP_ASSERT(orig.__invariants());
  LIBCPP_ASSERT(substr.__invariants());
  assert(substr == expected);
  assert(substr.get_allocator() == alloc);
  LIBCPP_ASSERT(is_string_asan_correct(orig));
  LIBCPP_ASSERT(is_string_asan_correct(substr));
}

template <class S>
constexpr void test_string_pos_alloc(
    S orig, typename S::size_type pos, const typename S::allocator_type& alloc, should_throw_exception_t) {
#ifndef TEST_HAS_NO_EXCEPTIONS
  if (!std::is_constant_evaluated()) {
    try {
      [[maybe_unused]] S substr = S(std::move(orig), pos, alloc);
      assert(false);
    } catch (const std::out_of_range&) {
    }
  }
#else
  (void)orig;
  (void)pos;
  (void)alloc;
#endif
}

template <class S>
constexpr void test_string_pos_n(S orig, typename S::size_type pos, typename S::size_type n, S expected) {
#ifdef _LIBCPP_VERSION
  ConstexprDisableAllocationGuard g;
#endif
  S substr(std::move(orig), pos, n);
  LIBCPP_ASSERT(orig.__invariants());
  LIBCPP_ASSERT(orig.empty());
  LIBCPP_ASSERT(substr.__invariants());
  assert(substr == expected);
  LIBCPP_ASSERT(is_string_asan_correct(orig));
  LIBCPP_ASSERT(is_string_asan_correct(substr));
}

template <class S>
constexpr void test_string_pos_n(S orig, typename S::size_type pos, typename S::size_type n, should_throw_exception_t) {
#ifndef TEST_HAS_NO_EXCEPTIONS
  if (!std::is_constant_evaluated()) {
    try {
      [[maybe_unused]] S substr = S(std::move(orig), pos, n);
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
constexpr void test_string_pos_n_alloc(
    S orig, typename S::size_type pos, typename S::size_type n, const typename S::allocator_type& alloc, S expected) {
  S substr(std::move(orig), pos, n, alloc);
  LIBCPP_ASSERT(orig.__invariants());
  LIBCPP_ASSERT(substr.__invariants());
  assert(substr == expected);
  assert(substr.get_allocator() == alloc);
  LIBCPP_ASSERT(is_string_asan_correct(orig));
  LIBCPP_ASSERT(is_string_asan_correct(substr));
}

template <class S>
constexpr void test_string_pos_n_alloc(
    S orig,
    typename S::size_type pos,
    typename S::size_type n,
    const typename S::allocator_type& alloc,
    should_throw_exception_t) {
#ifndef TEST_HAS_NO_EXCEPTIONS
  if (!std::is_constant_evaluated()) {
    try {
      [[maybe_unused]] S substr = S(std::move(orig), pos, n, alloc);
      assert(false);
    } catch (const std::out_of_range&) {
    }
  }
#else
  (void)orig;
  (void)pos;
  (void)n;
  (void)alloc;
#endif
}

template <class S>
constexpr void test_string(const typename S::allocator_type& alloc) {
  test_string_pos<S>(STR(""), 0, STR(""));
  test_string_pos<S>(STR(""), 1, should_throw_exception);
  test_string_pos<S>(STR("Banane"), 1, STR("anane"));
  test_string_pos<S>(STR("Banane"), 6, STR(""));
  test_string_pos<S>(STR("Banane"), 7, should_throw_exception);
  test_string_pos<S>(STR("long long string so no SSO"), 0, STR("long long string so no SSO"));
  test_string_pos<S>(STR("long long string so no SSO"), 10, STR("string so no SSO"));
  test_string_pos<S>(STR("long long string so no SSO"), 26, STR(""));
  test_string_pos<S>(STR("long long string so no SSO"), 27, should_throw_exception);

  test_string_pos_alloc<S>(STR(""), 0, alloc, STR(""));
  test_string_pos_alloc<S>(STR(""), 1, alloc, should_throw_exception);
  test_string_pos_alloc<S>(STR("Banane"), 1, alloc, STR("anane"));
  test_string_pos_alloc<S>(STR("Banane"), 6, alloc, STR(""));
  test_string_pos_alloc<S>(STR("Banane"), 7, alloc, should_throw_exception);
  test_string_pos_alloc<S>(STR("long long string so no SSO"), 0, alloc, STR("long long string so no SSO"));
  test_string_pos_alloc<S>(STR("long long string so no SSO"), 10, alloc, STR("string so no SSO"));
  test_string_pos_alloc<S>(STR("long long string so no SSO"), 26, alloc, STR(""));
  test_string_pos_alloc<S>(STR("long long string so no SSO"), 27, alloc, should_throw_exception);

  test_string_pos_n<S>(STR(""), 0, 0, STR(""));
  test_string_pos_n<S>(STR(""), 0, 1, STR(""));
  test_string_pos_n<S>(STR(""), 1, 0, should_throw_exception);
  test_string_pos_n<S>(STR(""), 1, 1, should_throw_exception);
  test_string_pos_n<S>(STR("Banane"), 1, 10, STR("anane"));
  test_string_pos_n<S>(STR("Banane"), 6, 0, STR(""));
  test_string_pos_n<S>(STR("Banane"), 6, 5, STR(""));
  test_string_pos_n<S>(STR("Banane"), 7, 10, should_throw_exception);
  test_string_pos_n<S>(STR("long long string so no SSO"), 0, 10, STR("long long "));
  test_string_pos_n<S>(STR("long long string so no SSO"), 10, 8, STR("string s"));
  test_string_pos_n<S>(STR("long long string so no SSO"), 20, 10, STR("no SSO"));
  test_string_pos_n<S>(STR("long long string so no SSO"), 26, 10, STR(""));
  test_string_pos_n<S>(STR("long long string so no SSO"), 27, 10, should_throw_exception);

  test_string_pos_n_alloc<S>(STR(""), 0, 0, alloc, STR(""));
  test_string_pos_n_alloc<S>(STR(""), 0, 1, alloc, STR(""));
  test_string_pos_n_alloc<S>(STR(""), 1, 0, alloc, should_throw_exception);
  test_string_pos_n_alloc<S>(STR(""), 1, 1, alloc, should_throw_exception);
  test_string_pos_n_alloc<S>(STR("Banane"), 1, 10, alloc, STR("anane"));
  test_string_pos_n_alloc<S>(STR("Banane"), 6, 0, alloc, STR(""));
  test_string_pos_n_alloc<S>(STR("Banane"), 6, 5, alloc, STR(""));
  test_string_pos_n_alloc<S>(STR("Banane"), 7, 10, alloc, should_throw_exception);
  test_string_pos_n_alloc<S>(STR("long long string so no SSO"), 0, 10, alloc, STR("long long "));
  test_string_pos_n_alloc<S>(STR("long long string so no SSO"), 10, 8, alloc, STR("string s"));
  test_string_pos_n_alloc<S>(STR("long long string so no SSO"), 20, 10, alloc, STR("no SSO"));
  test_string_pos_n_alloc<S>(STR("long long string so no SSO"), 26, 10, alloc, STR(""));
  test_string_pos_n_alloc<S>(STR("long long string so no SSO"), 27, 10, alloc, should_throw_exception);
}

template <class CharT, class CharTraits>
constexpr void test_allocators() {
  test_string<std::basic_string<CharT, CharTraits, std::allocator<CharT>>>(std::allocator<CharT>{});
  test_string<std::basic_string<CharT, CharTraits, min_allocator<CharT>>>(min_allocator<CharT>{});
  test_string<std::basic_string<CharT, CharTraits, test_allocator<CharT>>>(test_allocator<CharT>{42});
}

template <class CharT>
constexpr bool test_char_traits() {
  test_allocators<CharT, std::char_traits<CharT>>();
  test_allocators<CharT, constexpr_char_traits<CharT>>();

  return true;
}

int main(int, char**) {
  // TODO: put these into a single function when we increase the constexpr step limit
  test_char_traits<char>();
  static_assert(test_char_traits<char>());
  test_char_traits<char16_t>();
  static_assert(test_char_traits<char16_t>());
  test_char_traits<char32_t>();
  static_assert(test_char_traits<char32_t>());
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_char_traits<wchar_t>();
  static_assert(test_char_traits<wchar_t>());
#endif
#ifndef TEST_HAS_NO_CHAR8_T
  test_char_traits<char8_t>();
  static_assert(test_char_traits<char8_t>());
#endif

  return 0;
}
