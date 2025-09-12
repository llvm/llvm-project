//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

#include <deque>

#include "../../from_range_sequence_containers.h"
#include "test_macros.h"

// template<container-compatible-range<T> R>
//   deque(from_range_t, R&& rg, const Allocator& = Allocator()); // C++23

int main(int, char**) {
  for_all_iterators_and_allocators<int>([]<class Iter, class Sent, class Alloc>() {
    test_sequence_container<std::deque, int, Iter, Sent, Alloc>([]([[maybe_unused]] const auto& c) {
      LIBCPP_ASSERT(c.__invariants());
    });
  });
  test_sequence_container_move_only<std::deque>();

  static_assert(test_constraints<std::deque, int, double>());

  // TODO(varconst): `deque`'s constructors currently aren't exception-safe.
  // See https://llvm.org/PR62056.
  //test_exception_safety_throwing_copy<std::deque>();
  //test_exception_safety_throwing_allocator<std::deque, int>();

  return 0;
}
