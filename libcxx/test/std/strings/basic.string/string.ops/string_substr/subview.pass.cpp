//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <string>

// constexpr basic_string_view<_CharT, _Traits> subview(size_type __pos = 0, size_type __n = npos) const;

#include <cassert>
#include <string>

#include "asan_testing.h"
#include "constexpr_char_traits.h"
#include "make_string.h"
#include "min_allocator.h"
#include "test_allocator.h"
#include "test_macros.h"

#define CS(S) MAKE_CSTRING(CharT, S)

template <typename CharT, typename TraitsT, typename AllocT>
constexpr void test() {
  std::basic_string<CharT, TraitsT, AllocT> s{CS("Hello cruel world!"), AllocT{}};

  std::same_as<std::basic_string_view<CharT, TraitsT>> decltype(auto) sv = s.subview(6);
  assert(sv == CS("cruel world!"));

  std::same_as<std::basic_string_view<CharT, TraitsT>> decltype(auto) subsv = sv.subview(0, 5);
  assert(subsv == CS("cruel"));
}

template <typename CharT>
constexpr void test() {
  test<CharT, std::char_traits<CharT>, std::allocator<CharT>>();
  test<CharT, std::char_traits<CharT>, min_allocator<CharT>>();
  test<CharT, std::char_traits<CharT>, safe_allocator<CharT>>();
  test<CharT, std::char_traits<CharT>, test_allocator<CharT>>();

  test<CharT, constexpr_char_traits<CharT>, std::allocator<CharT>>();
  test<CharT, constexpr_char_traits<CharT>, min_allocator<CharT>>();
  test<CharT, constexpr_char_traits<CharT>, safe_allocator<CharT>>();
  test<CharT, constexpr_char_traits<CharT>, test_allocator<CharT>>();
}

constexpr bool test() {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif
#ifndef TEST_HAS_NO_CHAR8_T
  test<char8_t>();
#endif
  test<char16_t>();
  test<char32_t>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
