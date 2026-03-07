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

// Trying to advance chunk_view iterator out of range

#include <iterator>
#include <ranges>
#include <vector>

#include "check_assertion.h"

int main() {
  std::vector<int> vector = {1, 2, 3, 4, 5, 6, 7, 8};
  auto chunked            = vector | std::views::chunk(3);

  // Test advance past-the-end iterator when V models forward_range
  {
    /*chunk_view::__iterator*/ std::random_access_iterator auto it = chunked.begin();
    TEST_LIBCPP_ASSERT_FAILURE(it += 4, "Trying to advance chunk_view iterator out of range");
  }
}
