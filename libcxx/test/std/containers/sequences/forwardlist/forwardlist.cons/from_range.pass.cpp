//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<container-compatible-range<T> R>
//   forward_list(from_range_t, R&& rg, const Allocator& = Allocator()); // C++23; constexpr since C++26

#include <forward_list>

#include "../../from_range_sequence_containers.h"
#include "test_macros.h"

TEST_CONSTEXPR_CXX26 bool test() {
  for_all_iterators_and_allocators<int>([]<class Iter, class Sent, class Alloc>() {
    test_sequence_container<std::forward_list, int, Iter, Sent, Alloc>([](const auto&) {
      // No additional validation to do.
    });
  });
  test_sequence_container_move_only<std::forward_list>();

  static_assert(test_constraints<std::forward_list, int, double>());

  if (!TEST_IS_CONSTANT_EVALUATED) {
    test_exception_safety_throwing_copy<std::forward_list>();
    test_exception_safety_throwing_allocator<std::forward_list, int>();
  }

  return true;
}

int main(int, char**) {
  assert(test());
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}
