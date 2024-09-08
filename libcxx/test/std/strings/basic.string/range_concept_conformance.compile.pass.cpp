//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// string

#include <string>

#include <concepts>
#include <ranges>

#include "nasty_string.h"
#include "test_macros.h"

template <class String>
void test() {
  static_assert(std::same_as<std::ranges::iterator_t<String>, typename String::iterator>);
  static_assert(std::ranges::common_range<String>);
  static_assert(std::ranges::random_access_range<String>);
  static_assert(std::ranges::contiguous_range<String>);
  static_assert(!std::ranges::view<String>);
  static_assert(std::ranges::sized_range<String>);
  static_assert(!std::ranges::borrowed_range<String>);
  static_assert(std::ranges::viewable_range<String>);

  static_assert(std::same_as<std::ranges::iterator_t<String const>, typename String::const_iterator>);
  static_assert(std::ranges::common_range<String const>);
  static_assert(std::ranges::random_access_range<String const>);
  static_assert(std::ranges::contiguous_range<String const>);
  static_assert(!std::ranges::view<String const>);
  static_assert(std::ranges::sized_range<String const>);
  static_assert(!std::ranges::borrowed_range<String const>);
  static_assert(!std::ranges::viewable_range<String const>);
}

void tests() {
  test<std::string>();
  test<std::wstring>();
#ifndef TEST_HAS_NO_NASTY_STRING
  test<nasty_string>();
#endif
}
