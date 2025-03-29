//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <string>

// template<class Operation>
// void resize_and_overwrite(size_type n, Operation op)

#include <algorithm>
#include <cassert>
#include <memory>
#include <string>

#include "make_string.h"
#include "test_macros.h"
#include "asan_testing.h"

template <class S>
constexpr void test_appending(std::size_t k, size_t N, size_t new_capacity) {
  assert(N > k);
  assert(new_capacity >= N);
  auto s = S(k, 'a');
  s.resize_and_overwrite(new_capacity, [&](auto* p, auto n) {
    assert(n == new_capacity);
    LIBCPP_ASSERT(s.size() == new_capacity);
    LIBCPP_ASSERT(std::to_address(s.begin()) == p);
    assert(std::all_of(p, p + k, [](const auto ch) { return ch == 'a'; }));
    std::fill(p + k, p + n, 'b');
    p[n] = 'c'; // will be overwritten
    return N;
  });
  const S expected = S(k, 'a') + S(N - k, 'b');
  assert(s == expected);
  assert(s.c_str()[N] == '\0');
  LIBCPP_ASSERT(is_string_asan_correct(s));
}

template <class S>
constexpr void test_truncating(std::size_t o, size_t N) {
  assert(N < o);
  auto s = S(o, 'a');
  s.resize_and_overwrite(N, [&](auto* p, auto n) {
    assert(n == N);
    LIBCPP_ASSERT(s.size() == n);
    LIBCPP_ASSERT(std::to_address(s.begin()) == p);
    assert(std::all_of(p, p + n, [](auto ch) { return ch == 'a'; }));
    p[n - 1] = 'b';
    p[n]     = 'c'; // will be overwritten
    return n;
  });
  const S expected = S(N - 1, 'a') + S(1, 'b');
  assert(s == expected);
  assert(s.c_str()[N] == '\0');
  LIBCPP_ASSERT(is_string_asan_correct(s));
}

template <class String>
constexpr bool test() {
  test_appending<String>(10, 15, 15);
  test_appending<String>(10, 15, 20);
  test_appending<String>(10, 40, 40);
  test_appending<String>(10, 40, 50);
  test_appending<String>(30, 35, 35);
  test_appending<String>(30, 35, 45);
  test_appending<String>(10, 15, 30);
  test_truncating<String>(15, 10);
  test_truncating<String>(40, 35);
  test_truncating<String>(40, 10);

  return true;
}

void test_value_categories() {
  std::string s;
  s.resize_and_overwrite(10, [](char*&&, std::size_t&&) { return 0; });
  LIBCPP_ASSERT(is_string_asan_correct(s));
  s.resize_and_overwrite(10, [](char* const&, const std::size_t&) { return 0; });
  LIBCPP_ASSERT(is_string_asan_correct(s));
  struct RefQualified {
    int operator()(char*, std::size_t) && { return 0; }
  };
  s.resize_and_overwrite(10, RefQualified{});
  LIBCPP_ASSERT(is_string_asan_correct(s));
}

int main(int, char**) {
  test<std::basic_string<char, std::char_traits<char>, std::allocator<char>>>();
  test<std::basic_string<char8_t, std::char_traits<char8_t>, std::allocator<char8_t>>>();
  test<std::basic_string<char16_t, std::char_traits<char16_t>, std::allocator<char16_t>>>();
  test<std::basic_string<char32_t, std::char_traits<char32_t>, std::allocator<char32_t>>>();

  static_assert(test<std::basic_string<char, std::char_traits<char>, std::allocator<char>>>());
  static_assert(test<std::basic_string<char8_t, std::char_traits<char8_t>, std::allocator<char8_t>>>());
  static_assert(test<std::basic_string<char16_t, std::char_traits<char16_t>, std::allocator<char16_t>>>());
  static_assert(test<std::basic_string<char32_t, std::char_traits<char32_t>, std::allocator<char32_t>>>());

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<std::basic_string<wchar_t, std::char_traits<wchar_t>, std::allocator<wchar_t>>>();
  static_assert(test<std::basic_string<wchar_t, std::char_traits<wchar_t>, std::allocator<wchar_t>>>());
#endif
  return 0;
}
