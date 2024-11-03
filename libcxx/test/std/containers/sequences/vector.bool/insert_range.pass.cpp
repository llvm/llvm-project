//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-steps): -fconstexpr-steps=2000000

// template<container-compatible-range<bool> R>
//   constexpr iterator insert_range(const_iterator position, R&& rg); // C++23

#include <vector>

#include "../insert_range_sequence_containers.h"
#include "test_macros.h"

// Tested cases:
// - different kinds of insertions (inserting an {empty/one-element/mid-sized/long range} into an
//   {empty/one-element/full} container at the {beginning/middle/end});
// - an exception is thrown when allocating new elements.
constexpr bool test() {
  static_assert(test_constraints_insert_range<std::vector, bool, char>());

  for_all_iterators_and_allocators<bool, const int*>([]<class Iter, class Sent, class Alloc>() {
    test_sequence_insert_range<std::vector<bool, Alloc>, Iter, Sent>([](auto&& c) {
      LIBCPP_ASSERT(c.__invariants());
      // `is_contiguous_container_asan_correct` doesn't work on `vector<bool>`.
    });
  });

  { // Vector may or may not need to reallocate because of the insertion -- make sure to test both cases.
    { // Ensure reallocation happens.
      constexpr int N = 255;
      bool in[N] = {};
      std::vector<bool> v = {0, 0, 0, 1, 1, 0, 0, 0};
      auto initial = v;
      assert(v.capacity() < v.size() + std::ranges::size(in));

      v.insert_range(v.end(), in);
      // Because `in` is very large (it has to be to exceed the large capacity that `vector<bool>` allocates), it is
      // impractical to have the expected value as a literal.
      assert(v.size() == initial.size() + N);
      assert(std::ranges::equal(v.begin(), v.begin() + initial.size(), initial.begin(), initial.end()));
      assert(std::ranges::equal(v.begin() + initial.size(), v.end(), std::ranges::begin(in), std::ranges::end(in)));
    }

    { // Ensure no reallocation happens.
      bool in[] = {1, 1, 1, 1, 0, 0, 1, 1, 1, 1};
      std::vector<bool> v = {0, 0, 0, 1, 1, 0, 0, 0};
      v.reserve(v.size() + std::ranges::size(in));
      assert(v.capacity() >= v.size() + std::ranges::size(in));

      v.insert_range(v.end(), in);
      assert(std::ranges::equal(v, std::vector<bool>{0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1}));
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  // Note: `test_insert_range_exception_safety_throwing_copy` doesn't apply because copying booleans cannot throw.
  test_insert_range_exception_safety_throwing_allocator<std::vector, bool>();

  return 0;
}
