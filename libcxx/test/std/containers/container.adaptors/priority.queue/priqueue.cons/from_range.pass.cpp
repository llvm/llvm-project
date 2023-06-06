//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <queue>

#include "../../from_range_container_adaptors.h"
#include "../../../test_compare.h"
#include "test_macros.h"

// template <container-compatible-range<T> R>
//   priority_queue(from_range_t, R&& rg, const Compare& x = Compare()); // since C++23
// template <container-compatible-range<T> R, class Alloc>
//   priority_queue(from_range_t, R&& rg, const Compare&, const Alloc&); // since C++23
// template <container-compatible-range<T> R, class Alloc>
//   priority_queue(from_range_t, R&& rg, const Alloc&); // since C++23

template <class Range>
concept PriorityQueueHasFromRangeCtr = requires (Range&& range) {
  std::priority_queue<int>(std::from_range, std::forward<Range>(range));
  std::priority_queue<int>(std::from_range, std::forward<Range>(range), std::less<int>());
  std::priority_queue<int>(std::from_range, std::forward<Range>(range), std::less<int>(), std::allocator<int>());
  std::priority_queue<int>(std::from_range, std::forward<Range>(range), std::allocator<int>());
};

constexpr bool test_constraints_priority_queue() {
  // Input range with the same value type.
  static_assert(PriorityQueueHasFromRangeCtr<InputRange<int>>);
  // Input range with a convertible value type.
  static_assert(PriorityQueueHasFromRangeCtr<InputRange<double>>);
  // Input range with a non-convertible value type.
  static_assert(!PriorityQueueHasFromRangeCtr<InputRange<Empty>>);
  // Not an input range.
  static_assert(!PriorityQueueHasFromRangeCtr<InputRangeNotDerivedFrom>);
  static_assert(!PriorityQueueHasFromRangeCtr<InputRangeNotIndirectlyReadable>);
  static_assert(!PriorityQueueHasFromRangeCtr<InputRangeNotInputOrOutputIterator>);

  return true;
}

int main(int, char**) {
  for_all_iterators_and_allocators<int>([]<class Iter, class Sent, class Alloc>() {
    test_priority_queue<std::vector, int, Iter, Sent, test_less<int>, Alloc>();
  });
  test_container_adaptor_move_only<std::priority_queue>();

  static_assert(test_constraints_priority_queue());

  test_exception_safety_throwing_copy<std::priority_queue>();
  test_exception_safety_throwing_allocator<std::priority_queue, int>();

  return 0;
}
