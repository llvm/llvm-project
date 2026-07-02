//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// FIXME: This takes over an hour to compile, disable for now.
// UNSUPPORTED: LIBCXX-AMDGPU-FIXME
// UNSUPPORTED: LIBCXX-NVPTX-FIXME

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// template<container-compatible-range<T> R>
//   constexpr iterator insert_range(const_iterator position, R&& rg); // C++23; constexpr since C++26

#include <cassert>
#include <deque>

#include "../../insert_range_sequence_containers.h"
#include "test_macros.h"

// Tested cases:
// - different kinds of insertions (inserting an {empty/one-element/mid-sized/long range} into an
//   {empty/one-element/full} container at the {beginning/middle/end});
// - inserting move-only elements;
// - an exception is thrown when copying the elements or when allocating new elements.

#if TEST_STD_VER >= 26
constexpr bool test() {
  int input[]       = {2, 3};
  std::deque<int> d = {1, 4};
  auto it           = d.insert_range(d.begin() + 1, input);
  assert(*it == 2);
  assert((d == std::deque<int>{1, 2, 3, 4}));
  return true;
}
#endif

int main(int, char**) {
#if TEST_STD_VER >= 26
  test();
  static_assert(test());
#endif

  static_assert(test_constraints_insert_range<std::deque, int, double>());

  for_all_iterators_and_allocators<int, const int*>([]<class Iter, class Sent, class Alloc>() {
    test_sequence_insert_range<std::deque<int, Alloc>, Iter, Sent>([]([[maybe_unused]] auto&& c) {
      LIBCPP_ASSERT(c.__invariants());
    });
  });
  test_sequence_insert_range_move_only<std::deque>();

  test_insert_range_exception_safety_throwing_copy<std::deque>();
  test_insert_range_exception_safety_throwing_allocator<std::deque, int>();

  return 0;
}
