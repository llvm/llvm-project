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

template<class It, class Sentinel, class CharT>
constexpr void test_ctad(std::basic_string_view<CharT> val) {
  auto sv = std::basic_string_view(It(val.data()), Sentinel(It(val.data() + val.size())));
  ASSERT_SAME_TYPE(decltype(sv), std::basic_string_view<CharT>);
  assert(sv.data() == val.data());
  assert(sv.size() == val.size());
}

template<class CharT>
constexpr void test_with_char() {
  const auto val = MAKE_STRING_VIEW(CharT, "test");
  test_ctad<CharT*, CharT*>(val);
  test_ctad<CharT*, const CharT*>(val);
  test_ctad<const CharT*, CharT*>(val);
  test_ctad<const CharT*, sized_sentinel<const CharT*>>(val);
  test_ctad<contiguous_iterator<const CharT*>, contiguous_iterator<const CharT*>>(val);
  test_ctad<contiguous_iterator<const CharT*>, sized_sentinel<contiguous_iterator<const CharT*>>>(val);
}

constexpr void test() {
  test_with_char<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_with_char<wchar_t>();
#endif
  test_with_char<char8_t>();
  test_with_char<char16_t>();
  test_with_char<char32_t>();
}

int main(int, char**) {
  test();

  return 0;
}
