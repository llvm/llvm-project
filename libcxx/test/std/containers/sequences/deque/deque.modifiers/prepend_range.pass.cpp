//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// template<container-compatible-range<T> R>
//   constexpr void prepend_range(R&& rg); // C++23

#include <deque>

#include "../../insert_range_sequence_containers.h"
#include "test_macros.h"

// Tested cases:
// - different kinds of insertions (prepending an {empty/one-element/mid-sized/long range} into an
//   {empty/one-element/full} container);
// - prepending move-only elements;
// - an exception is thrown when copying the elements or when allocating new elements.
int main(int, char**) {
  static_assert(test_constraints_prepend_range<std::deque, int, double>());

  for_all_iterators_and_allocators<int, const int*>([]<class Iter, class Sent, class Alloc>() {
    test_sequence_prepend_range<std::deque<int, Alloc>, Iter, Sent>([]([[maybe_unused]] auto&& c) {
      LIBCPP_ASSERT(c.__invariants());
    });
  });
  test_sequence_prepend_range_move_only<std::deque>();

  test_prepend_range_exception_safety_throwing_copy<std::deque>();
  test_prepend_range_exception_safety_throwing_allocator<std::deque, int>();

  return 0;
}
