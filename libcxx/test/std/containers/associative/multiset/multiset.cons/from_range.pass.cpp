//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-steps): -fconstexpr-steps=2000000

// template<container-compatible-range<value_type> R>
//   multiset(from_range_t, R&& rg, const Compare& comp = Compare(), const Allocator& = Allocator()); // C++23, constexpr since C++26
//
// template<container-compatible-range<value_type> R>
//   multiset(from_range_t, R&& rg, const Allocator& a))
//     : multiset(from_range, std::forward<R>(rg), Compare(), a) { } // C++23, constexpr since C++26

#include <algorithm>
#include <array>
#include <set>

#include "../../from_range_associative_containers.h"
#include "test_macros.h"

TEST_CONSTEXPR_CXX26 void test_duplicates() {
  std::array input = {1, 2, 3, 3, 3, 4, 2, 1, 2};
  auto c           = std::multiset<int>(std::from_range, input);
  assert(std::ranges::is_permutation(input, c));
}

TEST_CONSTEXPR_CXX26 bool test() {
  for_all_iterators_and_allocators<int>([]<class Iter, class Sent, class Alloc>() {
    test_associative_set<std::multiset, int, Iter, Sent, test_less<int>, Alloc>();
  });
  test_associative_set_move_only<std::multiset>();
  test_duplicates();

  static_assert(test_set_constraints<std::multiset, int, double>());

  test_set_exception_safety_throwing_copy<std::multiset>();
  test_set_exception_safety_throwing_allocator<std::multiset, int>();

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif
  return 0;
}
