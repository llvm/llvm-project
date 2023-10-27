//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<container-compatible-range<value_type> R>
//   unordered_multimap(from_range_t, R&& rg, size_type n = see below,
//     const hasher& hf = hasher(), const key_equal& eql = key_equal(),
//     const allocator_type& a = allocator_type()); // C++23
//
// template<container-compatible-range<value_type> R>
//   unordered_multimap(from_range_t, R&& rg, size_type n, const allocator_type& a)
//     : unordered_multimap(from_range, std::forward<R>(rg), n, hasher(), key_equal(), a) { } // C++23
//
// template<container-compatible-range<value_type> R>
//   unordered_multimap(from_range_t, R&& rg, size_type n, const hasher& hf, const allocator_type& a)
//     : unordered_multimap(from_range, std::forward<R>(rg), n, hf, key_equal(), a) { }       // C++23

#include <array>
#include <unordered_map>

#include "../../from_range_unordered_containers.h"
#include "test_macros.h"

void test_duplicates() {
  using T = std::pair<const int, char>;
  std::array input = {
    T{1, 'a'}, T{2, 'a'}, T{3, 'a'}, T{3, 'b'}, T{3, 'c'}, T{2, 'b'}, T{4, 'a'}
  };
  auto c = std::unordered_multimap<int, char>(std::from_range, input);
  assert(std::ranges::is_permutation(input, c));
}

int main(int, char**) {
  using T = std::pair<const int, int>;
  for_all_iterators_and_allocators<T>([]<class Iter, class Sent, class Alloc>() {
    test_unordered_map<std::unordered_multimap, int, int, Iter, Sent, test_hash<int>, test_equal_to<int>, Alloc>();
  });
  test_unordered_map_move_only<std::unordered_multimap>();
  test_duplicates();

  static_assert(test_map_constraints<std::unordered_multimap, int, int, double, double>());

  test_map_exception_safety_throwing_copy<std::unordered_multimap>();
  test_map_exception_safety_throwing_allocator<std::unordered_multimap, int, int>();

  return 0;
}
