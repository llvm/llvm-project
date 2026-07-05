//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <set>

// class multiset

// template<class Key, class Compare, class Allocator>
//   synth-three-way-result<Key> operator<=>(const multiset<Key, Compare, Allocator>& x,
//                                           const multiset<Key, Compare, Allocator>& y);

#include <cassert>
#include <set>

#include "test_container_comparisons.h"

int main(int, char**) {
  assert(test_ordered_set_container_spaceship<std::multiset>());
  // `std::multiset` is not constexpr, so no `static_assert` test here.
  return 0;
}
