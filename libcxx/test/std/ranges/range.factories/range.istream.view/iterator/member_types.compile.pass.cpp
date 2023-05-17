//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-localization
// UNSUPPORTED: c++03, c++11, c++14, c++17

// Member typedefs in istream_view<T>::<iterator>.

#include <istream>
#include <ranges>

#include "test_macros.h"

template <class T>
concept HasIterCategory = requires { typename T::iterator_category; };

struct MemberIteratorCategory {
  using iterator_category = std::input_iterator_tag;
};
static_assert(HasIterCategory<MemberIteratorCategory>);

template <class Val, class CharT>
void test() {
  using Iter = std::ranges::iterator_t<std::ranges::basic_istream_view<Val, CharT>>;
  static_assert(std::is_same_v<typename Iter::iterator_concept, std::input_iterator_tag>);
  static_assert(std::is_same_v<typename Iter::difference_type, std::ptrdiff_t>);
  static_assert(std::is_same_v<typename Iter::value_type, Val>);
  static_assert(!HasIterCategory<Iter>);
}

template <class CharT>
void testOne() {
  test<int, CharT>();
  test<long, CharT>();
  test<double, CharT>();
  test<CharT, CharT>();
}

void test() {
  testOne<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  testOne<wchar_t>();
#endif
}
