//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<container-compatible-range<T> R>
//   constexpr void append_range(R&& rg); // C++23; constexpr since C++26

#include <list>
#include <type_traits>

#include "../../insert_range_sequence_containers.h"
#include "test_macros.h"

// Tested cases:
// - different kinds of insertions (appending an {empty/one-element/mid-sized/long range} into an
//   {empty/one-element/full} container);
// - appending move-only elements;
// - an exception is thrown when copying the elements or when allocating new elements.
TEST_CONSTEXPR_CXX26 bool test() {
  static_assert(test_constraints_append_range<std::list, int, double>());

  for_all_iterators_and_allocators<int, const int*>([]<class Iter, class Sent, class Alloc>() {
    test_sequence_append_range<std::list<int, Alloc>, Iter, Sent>([](auto&&) {
      // No additional validation to do.
    });
  });
  test_sequence_append_range_move_only<std::list>();

  if (!std::is_constant_evaluated()) {
    test_append_range_exception_safety_throwing_copy<std::list>();
    test_append_range_exception_safety_throwing_allocator<std::list, int>();
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
