//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// REQUIRES: has-unix-headers
// REQUIRES: libcpp-hardening-mode={{extensive|debug}}
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

// <ranges>

// Construct a chunk_view with chunk size <= 0

#include <ranges>
#include <vector>

#include "check_assertion.h"
#include "test_iterators.h"

int main() {
  std::vector<int> vector = {1, 2, 3};

  // Test constructor when V models only input_range.
  {
    TEST_LIBCPP_ASSERT_FAILURE(
        (std::ranges::subrange<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>(
             cpp17_input_iterator<int*>(vector.data()),
             sentinel_wrapper<cpp17_input_iterator<int*>>(cpp17_input_iterator<int*>(vector.data() + vector.size()))) |
         std::views::chunk(0)),
        "Trying to construct a chunk_view with chunk size <= 0");
    TEST_LIBCPP_ASSERT_FAILURE(
        (std::ranges::subrange<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>(
             cpp17_input_iterator<int*>(vector.data()),
             sentinel_wrapper<cpp17_input_iterator<int*>>(cpp17_input_iterator<int*>(vector.data() + vector.size()))) |
         std::views::chunk(-1)),
        "Trying to construct a chunk_view with chunk size <= 0");
  }

  // Test constructor when V models forward_range.
  {
    TEST_LIBCPP_ASSERT_FAILURE(vector | std::views::chunk(0), "Trying to construct a chunk_view with chunk size <= 0");
    TEST_LIBCPP_ASSERT_FAILURE(vector | std::views::chunk(-1), "Trying to construct a chunk_view with chunk size <= 0");
  }
}
