//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: no-exceptions
// REQUIRES: has-unix-headers

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// check that std::fill(ExecutionPolicy) and std::fill_n(ExecutionPolicy) terminate on user-thrown exceptions

#include <algorithm>

#include "check_assertion.h"
#include "test_execution_policies.h"
#include "test_iterators.h"

#ifndef TEST_HAS_NO_EXCEPTIONS
struct ThrowOnCopy {
  ThrowOnCopy& operator=(const ThrowOnCopy&) { throw int{}; }
};
#endif

int main(int, char**) {
  ThrowOnCopy a[2]{};
  int b[2]{};

  test_execution_policies([&](auto&& policy) {
    // std::fill
    EXPECT_STD_TERMINATE([&] { (void)std::fill(policy, std::begin(a), std::end(a), ThrowOnCopy{}); });
    EXPECT_STD_TERMINATE([&] {
      try {
        (void)std::fill(
            policy, util::throw_on_move_iterator(std::begin(b), 1), util::throw_on_move_iterator(std::end(b), 1), 0);
      } catch (const util::iterator_error&) {
        assert(false);
      }
      std::terminate(); // make the test pass in case the algorithm didn't move the iterator
    });

    // std::fill_n
    EXPECT_STD_TERMINATE([&] { (void)std::fill_n(policy, std::begin(a), std::size(a), ThrowOnCopy{}); });
    EXPECT_STD_TERMINATE([&] {
      try {
        (void)std::fill_n(policy, util::throw_on_move_iterator(std::begin(b), 1), std::size(b), 0);
      } catch (const util::iterator_error&) {
        assert(false);
      }
      std::terminate(); // make the test pass in case the algorithm didn't move the iterator
    });
  });
}
