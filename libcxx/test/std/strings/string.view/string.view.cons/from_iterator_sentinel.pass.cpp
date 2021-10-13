//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// <string_view>

//  template <class It, class End>
//  constexpr basic_string_view(It begin, End end)

#include <string_view>
#include <cassert>
#include <ranges>

#include "make_string.h"
#include "test_iterators.h"

template<class CharT, class Sentinel>
constexpr void test() {
  auto val = MAKE_STRING_VIEW(CharT, "test");
  auto sv = std::basic_string_view<CharT>(val.begin(), Sentinel(val.end()));
  ASSERT_SAME_TYPE(decltype(sv), std::basic_string_view<CharT>);
  assert(sv.size() == val.size());
  assert(sv.data() == val.data());
}

constexpr bool test() {
  test<char, char*>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t, wchar_t*>();
#endif
  test<char8_t, char8_t*>();
  test<char16_t, char16_t*>();
  test<char32_t, char32_t*>();
  test<char, const char*>();
  test<char, sized_sentinel<const char*>>();
  return true;
}

static_assert( std::is_constructible_v<std::string_view, const char*, char*>);
static_assert( std::is_constructible_v<std::string_view, char*, const char*>);
static_assert(!std::is_constructible_v<std::string_view, char*, void*>);               // not a sentinel
static_assert(!std::is_constructible_v<std::string_view, signed char*, signed char*>); // wrong char type
static_assert(!std::is_constructible_v<std::string_view, random_access_iterator<char*>, random_access_iterator<char*>>); // not contiguous
static_assert( std::is_constructible_v<std::string_view, contiguous_iterator<char*>, contiguous_iterator<char*>>);

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

