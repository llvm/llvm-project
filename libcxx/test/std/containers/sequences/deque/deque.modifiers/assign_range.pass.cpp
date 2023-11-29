//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<container-compatible-range<T> R>
//   constexpr void assign_range(R&& rg); // C++23

#include <deque>

#include "../../insert_range_sequence_containers.h"
#include "test_macros.h"

// Tested cases:
// - different kinds of assignments (assigning an {empty/one-element/mid-sized/long range} to an
//   {empty/one-element/full} container);
// - assigning move-only elements;
// - an exception is thrown when copying the elements or when allocating new elements.
int main(int, char**) {
  static_assert(test_constraints_assign_range<std::deque, int, double>());

  for_all_iterators_and_allocators<int, const int*>([]<class Iter, class Sent, class Alloc>() {
    test_sequence_assign_range<std::deque<int, Alloc>, Iter, Sent>([]([[maybe_unused]] auto&& c) {
      LIBCPP_ASSERT(c.__invariants());
    });
  });
  test_sequence_assign_range_move_only<std::deque>();

  test_assign_range_exception_safety_throwing_copy<std::deque>();
  test_assign_range_exception_safety_throwing_allocator<std::deque, int>();

  return 0;
}
