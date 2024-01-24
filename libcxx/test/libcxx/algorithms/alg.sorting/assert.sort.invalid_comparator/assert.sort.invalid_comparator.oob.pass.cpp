//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03, c++11, c++14, c++17
// In the debug mode, the comparator validations will notice that it doesn't satisfy strict weak ordering before the
// algorithm actually runs and goes out of bounds, so the test will terminate before the tested assertions are
// triggered.
// UNSUPPORTED: libcpp-hardening-mode=none, libcpp-hardening-mode=debug
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "bad_comparator_values.h"
#include "check_assertion.h"
#include "invalid_comparator_utilities.h"

void check_oob_sort_read() {
  SortingFixture fixture(SORT_DATA);

  // Check the classic sorting algorithms
  {
    std::vector<std::size_t*> copy = fixture.create_elements();
    TEST_LIBCPP_ASSERT_FAILURE(
        std::sort(copy.begin(), copy.end(), fixture.checked_predicate()),
        "Would read out of bounds, does your comparator satisfy the strict-weak ordering requirement?");
  }

  // Check the Ranges sorting algorithms
  {
    std::vector<std::size_t*> copy = fixture.create_elements();
    TEST_LIBCPP_ASSERT_FAILURE(
        std::ranges::sort(copy, fixture.checked_predicate()),
        "Would read out of bounds, does your comparator satisfy the strict-weak ordering requirement?");
  }
}

void check_oob_nth_element_read() {
  SortingFixture fixture(NTH_ELEMENT_DATA);

  {
    std::vector<std::size_t*> copy = fixture.create_elements();
    TEST_LIBCPP_ASSERT_FAILURE(
        std::nth_element(copy.begin(), copy.begin(), copy.end(), fixture.checked_predicate()),
        "Would read out of bounds, does your comparator satisfy the strict-weak ordering requirement?");
  }

  {
    std::vector<std::size_t*> copy = fixture.create_elements();
    TEST_LIBCPP_ASSERT_FAILURE(
        std::ranges::nth_element(copy, copy.begin(), fixture.checked_predicate()),
        "Would read out of bounds, does your comparator satisfy the strict-weak ordering requirement?");
  }
}

int main(int, char**) {
  check_oob_sort_read();
  check_oob_nth_element_read();

  return 0;
}
