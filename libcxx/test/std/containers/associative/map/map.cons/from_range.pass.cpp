//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<container-compatible-range<value_type> R>
//   map(from_range_t, R&& rg, const Compare& comp = Compare(), const Allocator& = Allocator()); // C++23
//
// template<container-compatible-range<value_type> R>
//   map(from_range_t, R&& rg, const Allocator& a))
//     : map(from_range, std::forward<R>(rg), Compare(), a) { } // C++23

#include <array>
#include <map>

#include "../../from_range_associative_containers.h"
#include "test_macros.h"

void test_duplicates() {
  using T = std::pair<const int, char>;

  std::array input = {
    T{1, 'a'}, T{2, 'a'}, T{3, 'a'}, T{3, 'b'}, T{3, 'c'}, T{2, 'b'}, T{4, 'a'}
  };
  std::array expected = {
    T{1, 'a'}, T{2, 'a'}, T{3, 'a'}, T{4, 'a'}
  };
  auto c = std::map<int, char>(std::from_range, input);
  assert(std::ranges::is_permutation(expected, c));
}

int main(int, char**) {
  using T = std::pair<const int, int>;
  for_all_iterators_and_allocators<T>([]<class Iter, class Sent, class Alloc>() {
    test_associative_map<std::map, int, int, Iter, Sent, test_less<int>, Alloc>();
  });
  test_associative_map_move_only<std::map>();
  test_duplicates();

  static_assert(test_map_constraints<std::map, int, int, double, double>());

  test_map_exception_safety_throwing_copy<std::map>();
  test_map_exception_safety_throwing_allocator<std::map, int, int>();

  return 0;
}
