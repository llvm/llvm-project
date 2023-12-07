//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<container-compatible-range<value_type> R>
//   unordered_set(from_range_t, R&& rg, size_type n = see below,
//     const hasher& hf = hasher(), const key_equal& eql = key_equal(),
//     const allocator_type& a = allocator_type()); // C++23
//
// template<container-compatible-range<value_type> R>
//   unordered_set(from_range_t, R&& rg, size_type n, const allocator_type& a)
//     : unordered_set(from_range, std::forward<R>(rg), n, hasher(), key_equal(), a) { } // C++23
//
// template<container-compatible-range<value_type> R>
//   unordered_set(from_range_t, R&& rg, size_type n, const hasher& hf, const allocator_type& a)
//     : unordered_set(from_range, std::forward<R>(rg), n, hf, key_equal(), a) { }       // C++23

#include <array>
#include <unordered_set>

#include "../../from_range_unordered_containers.h"
#include "test_macros.h"

void test_duplicates() {
  using T = KeyValue;

  std::array input = {
    T{1, 'a'}, T{2, 'a'}, T{3, 'a'}, T{3, 'b'}, T{3, 'c'}, T{2, 'b'}, T{4, 'a'}
  };
  std::array expected = {
    T{1, 'a'}, T{2, 'b'}, T{3, 'c'}, T{4, 'a'}
  };
  auto c = std::unordered_set<T>(std::from_range, input);
  assert(std::ranges::is_permutation(expected, c));
}

int main(int, char**) {
  for_all_iterators_and_allocators<int>([]<class Iter, class Sent, class Alloc>() {
    test_unordered_set<std::unordered_set, int, Iter, Sent, test_hash<int>, test_equal_to<int>, Alloc>();
  });
  test_unordered_set_move_only<std::unordered_set>();
  test_duplicates();

  static_assert(test_set_constraints<std::unordered_set, int, double>());

  test_set_exception_safety_throwing_copy<std::unordered_set>();
  test_set_exception_safety_throwing_allocator<std::unordered_set, int>();

  return 0;
}
