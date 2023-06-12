//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<container-compatible-range<T> R>
//   void push_range(R&& rg); // C++23

#include <queue>

#include "../../push_range_container_adaptors.h"
#include "test_macros.h"

int main(int, char**) {
  for_all_iterators_and_allocators<int, const int*>([]<class Iter, class Sent, class Alloc>() {
    test_push_range<std::priority_queue<int, std::vector<int, Alloc>>, Iter, Sent>(/*is_result_heapified=*/true);
  });
  test_push_range_move_only<std::priority_queue>();
  test_push_range_inserter_choice<std::priority_queue, int>(/*is_result_heapified=*/true);

  static_assert(test_constraints_push_range<std::priority_queue, int, double>());

  test_push_range_exception_safety_throwing_copy<std::priority_queue>();
  test_push_range_exception_safety_throwing_allocator<std::priority_queue, std::vector, int>();

  return 0;
}

