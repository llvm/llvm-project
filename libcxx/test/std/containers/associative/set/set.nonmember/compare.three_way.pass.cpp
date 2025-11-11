//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <set>

// class set

// template<class Key, class Compare, class Allocator>
//   constexpr synth-three-way-result<Key> operator<=>(const set<Key, Compare, Allocator>& x,
//                                           const set<Key, Compare, Allocator>& y); // constexpr since C++26

#include <cassert>
#include <set>

#include "test_container_comparisons.h"

TEST_CONSTEXPR_CXX26 bool test() {
  assert((test_ordered_set_container_spaceship<std::set>()));
  // `std::set` is not constexpr, so no `static_assert` test here.
  return true;
}
int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif
  return 0;
}
