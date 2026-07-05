//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<container-compatible-range<T> R>
//   constexpr void prepend_range(R&& rg); // C++23; constexpr since C++26

#include <forward_list>

#include "../../insert_range_sequence_containers.h"
#include "test_macros.h"

// Tested cases:
// - different kinds of insertions (prepending an {empty/one-element/mid-sized/long range} into an
//   {empty/one-element/full} container);
// - prepending move-only elements;
// - an exception is thrown when copying the elements or when allocating new elements.
TEST_CONSTEXPR_CXX26 bool test() {
  static_assert(test_constraints_assign_range<std::forward_list, int, double>());

  for_all_iterators_and_allocators<int, const int*>([]<class Iter, class Sent, class Alloc>() {
    test_sequence_prepend_range<std::forward_list<int, Alloc>, Iter, Sent>([](auto&&) {
      // No additional validation to do.
    });
  });
  test_sequence_prepend_range_move_only<std::forward_list>();

  if (!TEST_IS_CONSTANT_EVALUATED) {
    test_prepend_range_exception_safety_throwing_copy<std::forward_list>();
    test_prepend_range_exception_safety_throwing_allocator<std::forward_list, int>();
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
