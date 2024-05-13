//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <vector>

#include "../from_range_sequence_containers.h"
#include "test_macros.h"

// template<container-compatible-range<T> R>
//   vector(from_range_t, R&& rg, const Allocator& = Allocator()); // C++23

constexpr bool test() {
  for_all_iterators_and_allocators<bool>([]<class Iter, class Sent, class Alloc>() {
    test_vector_bool<Iter, Sent, Alloc>([]([[maybe_unused]] const auto& c) {
      LIBCPP_ASSERT(c.__invariants());
      // `is_contiguous_container_asan_correct` doesn't work on `vector<bool>`.
    });
  });

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  static_assert(test_constraints<std::vector, bool, char>());

  // Note: test_exception_safety_throwing_copy doesn't apply because copying a boolean cannot throw.
  test_exception_safety_throwing_allocator<std::vector, bool>();

  return 0;
}
