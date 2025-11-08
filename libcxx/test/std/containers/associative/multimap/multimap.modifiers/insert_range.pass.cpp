//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// Some fields in the test case variables are deliberately not explicitly initialized, this silences a warning on GCC.
// ADDITIONAL_COMPILE_FLAGS(gcc-style-warnings): -Wno-missing-field-initializers
// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-steps): -fconstexpr-steps=2147483647
// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-ops-limit): -fconstexpr-ops-limit=800000000

// <map>

// template<container-compatible-range<value_type> R>
//   void insert_range(R&& rg); // C++23 // constexpr since C++26

#include <map>

#include "../../../insert_range_maps_sets.h"
#include "test_macros.h"

TEST_CONSTEXPR_CXX26
bool test() {
  // Note: we want to use a pair with non-const elements for input (an assignable type is a lot more convenient) but
  // have to use the exact `value_type` of the map (that is, `pair<const K, V>`) for the allocator.
  using Pair      = std::pair<int, char>;
  using ConstPair = std::pair<const int, char>;
  for_all_iterators_and_allocators<ConstPair, const Pair*>([]<class Iter, class Sent, class Alloc>() {
    test_map_set_insert_range<std::multimap<int, char, test_less<int>, Alloc>, Pair, Iter, Sent>(
        /*allow_duplicates=*/true);
  });

  static_assert(test_map_constraints_insert_range<std::multimap, int, int, char, double>());

  test_map_insert_range_move_only<std::multimap>();

  test_map_insert_range_exception_safety_throwing_copy<std::multimap>();
  test_assoc_map_insert_range_exception_safety_throwing_allocator<std::multimap, int, int>();

  return true;
}
int main(int, char**) {
  test();

#if TEST_STD_VER >= 26
  static_assert(test());
#endif
  return 0;
}
