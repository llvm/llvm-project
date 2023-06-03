//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <stack>

#include "../../from_range_container_adaptors.h"
#include "test_macros.h"

// template<container-compatible-range<T> R> stack(from_range_t, R&& rg); // since C++23
// template<container-compatible-range<T> R, class Alloc>
//   stack(from_range_t, R&& rg, const Alloc&); // since C++23

int main(int, char**) {
  for_all_iterators_and_allocators<int>([]<class Iter, class Sent, class Alloc>() {
    test_container_adaptor<std::stack, std::deque, int, Iter, Sent, Alloc>();
  });
  test_container_adaptor_move_only<std::stack>();

  static_assert(test_constraints<std::stack, int, double>());

  test_exception_safety_throwing_copy<std::stack>();
  test_exception_safety_throwing_allocator<std::stack, int>();

  return 0;
}
