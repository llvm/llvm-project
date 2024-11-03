//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// Some fields in the test case variables are deliberately not explicitly initialized, this silences a warning on GCC.
// ADDITIONAL_COMPILE_FLAGS: -Wno-missing-field-initializers

// <set>

// template<container-compatible-range<value_type> R>
//   void insert_range(R&& rg); // C++23

#include <set>

#include "../../insert_range_maps_sets.h"
#include "test_macros.h"

int main(int, char**) {
  for_all_iterators_and_allocators<int, const int*>([]<class Iter, class Sent, class Alloc>() {
    test_map_set_insert_range<std::multiset<int, test_less<int>, Alloc>, int, Iter, Sent>(/*allow_duplicates=*/true);
  });

  static_assert(test_set_constraints_insert_range<std::multiset, int, double>());

  test_set_insert_range_move_only<std::multiset>();

  test_set_insert_range_exception_safety_throwing_copy<std::multiset>();
  test_assoc_set_insert_range_exception_safety_throwing_allocator<std::multiset, int>();

  return 0;
}
