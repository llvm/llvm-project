//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// <string_view>

//  template <contiguous_iterator _It, sized_sentinel_for<_It> _End>
//    basic_string_view(_It, _End) -> basic_string_view<iter_value_t<_It>>;

#include <string_view>
#include <cassert>

#include "make_string.h"
#include "test_macros.h"
#include "test_iterators.h"

template<class CharT, class Sentinel>
constexpr void test() {
  auto val = MAKE_STRING_VIEW(CharT, "test");
  auto sv = std::basic_string_view(val.begin(), Sentinel(val.end()));
  ASSERT_SAME_TYPE(decltype(sv), std::basic_string_view<CharT>);
  assert(sv.size() == val.size());
  assert(sv.data() == val.data());
}

constexpr void test() {
  test<char, char*>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t, wchar_t*>();
#endif
  test<char8_t, char8_t*>();
  test<char16_t, char16_t*>();
  test<char32_t, char32_t*>();
  test<char, const char*>();
  test<char, sized_sentinel<const char*>>();
}

int main(int, char**) {
  test();

  return 0;
}

