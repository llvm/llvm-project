//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>

// template <class T>
// struct hash
// {
//     size_t operator()(T val) const;
// };

// Not very portable

#include <string>
#include <cassert>
#include <type_traits>

#include "test_macros.h"

template <class T>
void test() {
  typedef std::hash<T> H;
#if TEST_STD_VER <= 14
  static_assert((std::is_same<typename H::argument_type, T>::value), "");
  static_assert((std::is_same<typename H::result_type, std::size_t>::value), "");
#endif
  ASSERT_NOEXCEPT(H()(T()));

  H h;
  std::string g1 = "1234567890";
  std::string g2 = "1234567891";
  T s1(g1.begin(), g1.end());
  T s2(g2.begin(), g2.end());
  assert(h(s1) != h(s2));
}

int main(int, char**) {
  test<std::string>();
#ifndef TEST_HAS_NO_CHAR8_T
  test<std::u8string>();
#endif
  test<std::u16string>();
  test<std::u32string>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<std::wstring>();
#endif

  return 0;
}
