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

// Trying to increment past-the-end chunk_view iterator

#include <iterator>
#include <ranges>
#include <vector>

#include "check_assertion.h"
#include "../types.h"

int main() {
  std::vector<int> vector = {1, 2, 3, 4, 5, 6, 7, 8};
  auto chunked            = vector | std::views::chunk(3);
  auto input_chunked      = input_span(vector.data(), 8) | std::views::chunk(3);

  // Test increment past-the-end iterator when V models only input_range
  {
    /*chunk_view::__outer_iterator*/ std::input_iterator auto outer_it = input_chunked.begin();
    ++outer_it;
    ++outer_it;
    ++outer_it;
    TEST_LIBCPP_ASSERT_FAILURE(++outer_it, "Trying to increment past-the-end chunk_view iterator");

    /*chunk_view::__inner_iterator*/ std::input_iterator auto inner_it = (*input_chunked.begin()).begin();
    ++inner_it;
    ++inner_it;
    ++inner_it;
    TEST_LIBCPP_ASSERT_FAILURE(++inner_it, "Trying to increment past-the-end chunk_view iterator");
  }

  // Test increment past-the-end iterator when V models forward_range
  {
    /*chunk_view::__iterator*/ std::random_access_iterator auto it = chunked.begin();
    ++it;
    ++it;
    ++it;
    TEST_LIBCPP_ASSERT_FAILURE(++it, "Trying to increment past-the-end chunk_view iterator");
  }
}
