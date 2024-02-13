//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// <iterator>
//
// Note that begin and end are tested in libcxx/test/std/language.support/support.initlist/support.initlist.range/begin_end.pass.cpp
//
// template <class E> constexpr reverse_iterator<const E*> rbegin(initializer_list<E> il); // C++14, constexpr since C++17
// template <class E> constexpr reverse_iterator<const E*> rend(initializer_list<E> il);   // C++14, constexpr since C++17

#include <cassert>
#include <initializer_list>
#include <iterator>

#include "test_macros.h"

TEST_CONSTEXPR_CXX17 bool test() {
  std::initializer_list<int> il = {1, 2, 3};
  ASSERT_SAME_TYPE(decltype(std::rbegin(il)), std::reverse_iterator<const int*>);
  ASSERT_SAME_TYPE(decltype(std::rend(il)), std::reverse_iterator<const int*>);
  assert(std::rbegin(il).base() == il.end());
  assert(std::rend(il).base() == il.begin());

  const auto& cil = il;
  ASSERT_SAME_TYPE(decltype(std::rbegin(cil)), std::reverse_iterator<const int*>);
  ASSERT_SAME_TYPE(decltype(std::rend(cil)), std::reverse_iterator<const int*>);
  assert(std::rbegin(cil).base() == il.end());
  assert(std::rend(cil).base() == il.begin());
  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 17
  static_assert(test());
#endif

  return 0;
}
