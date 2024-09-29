//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <inplace_vector>

// template<container-compatible-range<T> R>
//   inplace_vector(from_range_t, R&& rg);

#include <inplace_vector>

#include "../../from_range_sequence_containers.h"
#include "test_macros.h"

// TODO: from_range_sequence_containers.h does not support inplace_vector well because it doesn't have an allocator and has a NTTP
#if 0
constexpr bool test() {
  for_all_iterators_and_allocators<int>([]<class Iter, class Sent, class Alloc>() {
    test_sequence_container<std::inplace_vector, int, Iter, Sent, Alloc>([](const auto&) {});
  });
  test_sequence_container_move_only<std::inplace_vector>();

  return true;
}
#endif

int main(int, char**) {
#if 0
  static_assert(test_constraints<std::inplace_vector, int, double>());
  test();

  static_assert(test());

  test_exception_safety_throwing_copy<std::inplace_vector>();

  return 0;
#endif
}
