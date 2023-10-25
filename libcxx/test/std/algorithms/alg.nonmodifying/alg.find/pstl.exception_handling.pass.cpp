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

// check that std::find(ExecutionPolicy), std::find_if(ExecutionPolicy) and std::find_if_not(ExecutionPolicy) terminate
// on user-thrown exceptions

#include <algorithm>

#include "check_assertion.h"
#include "test_execution_policies.h"
#include "test_iterators.h"

struct ThrowOnCompare {};

#ifndef TEST_HAS_NO_EXCEPTIONS
bool operator==(ThrowOnCompare, ThrowOnCompare) { throw int{}; }
#endif

int main(int, char**) {
  test_execution_policies([](auto&& policy) {
    // std::find
    EXPECT_STD_TERMINATE([&] {
      ThrowOnCompare a[2] = {};
      (void)std::find(policy, std::begin(a), std::end(a), ThrowOnCompare{});
    });
    EXPECT_STD_TERMINATE([&] {
      try {
        int a[] = {1, 2};
        (void)std::find(
            policy, util::throw_on_move_iterator(std::begin(a), 1), util::throw_on_move_iterator(std::end(a), 1), 0);
      } catch (const util::iterator_error&) {
        assert(false);
      }
      std::terminate(); // make the test pass in case the algorithm didn't move the iterator
    });

    // std::find_if
    EXPECT_STD_TERMINATE([&] {
      int a[] = {1, 2};
      (void)std::find_if(policy, std::begin(a), std::end(a), [](int) -> bool { throw int{}; });
    });
    EXPECT_STD_TERMINATE([&] {
      try {
        int a[] = {1, 2};
        (void)std::find_if(
            policy,
            util::throw_on_move_iterator(std::begin(a), 1),
            util::throw_on_move_iterator(std::end(a), 1),
            [](int) { return true; });
      } catch (const util::iterator_error&) {
        assert(false);
      }
      std::terminate(); // make the test pass in case the algorithm didn't move the iterator
    });

    // std::find_if_not
    EXPECT_STD_TERMINATE([&] {
      int a[] = {1, 2};
      (void)std::find_if_not(policy, std::begin(a), std::end(a), [](int) -> bool { throw int{}; });
    });
    EXPECT_STD_TERMINATE([&] {
      try {
        int a[] = {1, 2};
        (void)std::find_if_not(
            policy,
            util::throw_on_move_iterator(std::begin(a), 1),
            util::throw_on_move_iterator(std::end(a), 1),
            [](int) { return true; });
      } catch (const util::iterator_error&) {
        assert(false);
      }
      std::terminate(); // make the test pass in case the algorithm didn't move the iterator
    });
  });
}
