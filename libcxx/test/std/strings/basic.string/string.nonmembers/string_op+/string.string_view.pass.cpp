//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <string>

// [string.op.plus]
//
// template<class charT, class traits, class Allocator>
//   constexpr basic_string<charT, traits, Allocator>
//     operator+(const basic_string<charT, traits, Allocator>& lhs,
//               type_identity_t<basic_string_view<charT, traits>> rhs);                           // Since C++26
// template<class charT, class traits, class Allocator>
//   constexpr basic_string<charT, traits, Allocator>
//     operator+(basic_string<charT, traits, Allocator>&& lhs,
//               type_identity_t<basic_string_view<charT, traits>> rhs);                           // Since C++26
// template<class charT, class traits, class Allocator>
//   constexpr basic_string<charT, traits, Allocator>
//     operator+(type_identity_t<basic_string_view<charT, traits>> lhs,
//               const basic_string<charT, traits, Allocator>& rhs);                               // Since C++26
// template<class charT, class traits, class Allocator>
//   constexpr basic_string<charT, traits, Allocator>
//     operator+(type_identity_t<basic_string_view<charT, traits>> lhs,
//               basic_string<charT, traits, Allocator>&& rhs);                                    // Since C++26

#include <cassert>
#include <string>
#include <utility>

#include "asan_testing.h"
#include "constexpr_char_traits.h"
#include "make_string.h"
#include "min_allocator.h"
#include "test_macros.h"

#define CS(S) MAKE_CSTRING(CharT, S)
#define ST(S, a) std::basic_string<CharT, TraitsT, AllocT>(MAKE_CSTRING(CharT, S), MKSTR_LEN(CharT, S), a)
#define SV(S) std::basic_string_view<CharT, TraitsT>(MAKE_CSTRING(CharT, S), MKSTR_LEN(CharT, S))

template <typename CharT, typename TraitsT, typename AllocT>
constexpr void test() {
  AllocT allocator;
  std::basic_string<CharT, TraitsT, AllocT> st{ST("Hello", allocator)};
  std::basic_string_view<CharT, TraitsT> sv{SV("World")};

  assert(st + sv == ST("HelloWorld", allocator));
  assert(st + sv != ST("Hello World", allocator));
}

constexpr bool test() {
  test<char, std::char_traits<char>, min_allocator<char>>();
  test<char, constexpr_char_traits<char>, min_allocator<char>>();
  test<char, std::char_traits<char>, safe_allocator<char>>();
  test<char, constexpr_char_traits<char>, safe_allocator<char>>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t, std::char_traits<wchar_t>, min_allocator<wchar_t>>();
  test<wchar_t, constexpr_char_traits<wchar_t>, min_allocator<wchar_t>>();
  test<wchar_t, std::char_traits<wchar_t>, safe_allocator<wchar_t>>();
  test<wchar_t, constexpr_char_traits<wchar_t>, safe_allocator<wchar_t>>();
#endif

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
