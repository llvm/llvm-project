//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: !stdlib=libc++ && (c++03 || c++11 || c++14)

// <string_view>

// class iterator

#include <cassert>
#include <concepts>
#include <iterator>
#include <string_view>

#include "test_macros.h"
#include "make_string.h"

template <class CharT>
TEST_CONSTEXPR_CXX14 void test_type() {
  using C                  = std::basic_string_view<CharT>;
  typename C::iterator ii1 = typename C::iterator(), ii2 = typename C::iterator();
  typename C::iterator ii4       = ii1;
  typename C::const_iterator cii = typename C::const_iterator();
  assert(ii1 == ii2);
  assert(ii1 == ii4);
  assert(ii1 == cii);

  assert(!(ii1 != ii2));
  assert(!(ii1 != cii));

#if TEST_STD_VER >= 17
  C c = MAKE_STRING_VIEW(CharT, "abc");
  assert(c.begin() == std::begin(c));
  assert(c.rbegin() == std::rbegin(c));
  assert(c.cbegin() == std::cbegin(c));
  assert(c.crbegin() == std::crbegin(c));

  assert(c.end() == std::end(c));
  assert(c.rend() == std::rend(c));
  assert(c.cend() == std::cend(c));
  assert(c.crend() == std::crend(c));

  assert(std::begin(c) != std::end(c));
  assert(std::rbegin(c) != std::rend(c));
  assert(std::cbegin(c) != std::cend(c));
  assert(std::crbegin(c) != std::crend(c));
#endif

#if TEST_STD_VER >= 20
  // P1614 + LWG3352
  std::same_as<std::strong_ordering> decltype(auto) r1 = ii1 <=> ii2;
  assert(r1 == std::strong_ordering::equal);

  std::same_as<std::strong_ordering> decltype(auto) r2 = ii1 <=> ii2;
  assert(r2 == std::strong_ordering::equal);
#endif
}

TEST_CONSTEXPR_CXX14 bool test() {
  test_type<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_type<wchar_t>();
#endif
#ifndef TEST_HAS_NO_CHAR8_T
  test_type<char8_t>();
#endif
  test_type<char16_t>();
  test_type<char32_t>();

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 14
  static_assert(test(), "");
#endif

  return 0;
}
