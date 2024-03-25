//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<container-compatible-range<value_type> R>
//   set(from_range_t, R&& rg, const Compare& comp = Compare(), const Allocator& = Allocator()); // C++23
//
// template<container-compatible-range<value_type> R>
//   set(from_range_t, R&& rg, const Allocator& a))
//     : set(from_range, std::forward<R>(rg), Compare(), a) { } // C++23

#include <array>
#include <set>

#include "../../from_range_associative_containers.h"
#include "test_macros.h"

void test_duplicates() {
  using T = KeyValue;

  std::array input = {
    T{1, 'a'}, T{2, 'a'}, T{3, 'a'}, T{3, 'b'}, T{3, 'c'}, T{2, 'b'}, T{4, 'a'}
  };
  std::array expected = {
    T{1, 'a'}, T{2, 'b'}, T{3, 'c'}, T{4, 'a'}
  };
  auto c = std::set<T>(std::from_range, input);
  assert(std::ranges::is_permutation(expected, c));
}

int main(int, char**) {
  for_all_iterators_and_allocators<int>([]<class Iter, class Sent, class Alloc>() {
    test_associative_set<std::set, int, Iter, Sent, test_less<int>, Alloc>();
  });
  test_associative_set_move_only<std::set>();
  test_duplicates();

  static_assert(test_set_constraints<std::set, int, double>());

  test_set_exception_safety_throwing_copy<std::set>();
  test_set_exception_safety_throwing_allocator<std::set, int>();

  return 0;
}
