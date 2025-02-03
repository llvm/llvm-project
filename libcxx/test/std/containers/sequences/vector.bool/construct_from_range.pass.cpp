//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <sstream>
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

  { // Ensure input-only sized ranges are accepted.
    using input_iter = cpp20_input_iterator<const bool*>;
    const bool in[]{true, true, false, true};
    std::vector v(std::from_range, std::views::counted(input_iter{std::ranges::begin(in)}, std::ranges::ssize(in)));
    assert(std::ranges::equal(v, std::vector<bool>{true, true, false, true}));
  }

  return true;
}

#ifndef TEST_HAS_NO_LOCALIZATION
void test_counted_istream_view() {
  std::istringstream is{"1 1 0 1"};
  auto vals = std::views::istream<bool>(is);
  std::vector v(std::from_range, std::views::counted(vals.begin(), 3));
  assert(v == (std::vector{true, true, false}));
}
#endif

int main(int, char**) {
  test();
  static_assert(test());

  static_assert(test_constraints<std::vector, bool, char>());

  // Note: test_exception_safety_throwing_copy doesn't apply because copying a boolean cannot throw.
  test_exception_safety_throwing_allocator<std::vector, bool>();

#ifndef TEST_HAS_NO_LOCALIZATION
  test_counted_istream_view();
#endif

  return 0;
}
