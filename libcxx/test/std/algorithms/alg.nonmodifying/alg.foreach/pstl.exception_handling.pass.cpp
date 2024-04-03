//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: no-exceptions
// `check_assertion.h` requires Unix headers and regex support.
// UNSUPPORTED: !has-unix-headers, no-localization

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// check that std::for_each(ExecutionPolicy) and std::for_each_n(ExecutionPolicy) terminate on user-thrown exceptions

#include <algorithm>

#include "check_assertion.h"
#include "test_execution_policies.h"
#include "test_iterators.h"

int main(int, char**) {
  test_execution_policies([](auto&& policy) {
    int a[] = {1, 2};
    // std::for_each
    EXPECT_STD_TERMINATE([&] { std::for_each(policy, std::begin(a), std::end(a), [](int) { throw int{}; }); });
    EXPECT_STD_TERMINATE([&] {
      try {
        (void)std::for_each(
            policy,
            util::throw_on_move_iterator(std::begin(a), 1),
            util::throw_on_move_iterator(std::end(a), 1),
            [](int) {});
      } catch (const util::iterator_error&) {
        assert(false);
      }
      std::terminate(); // make the test pass in case the algorithm didn't move the iterator
    });

    // std::for_each_n
    EXPECT_STD_TERMINATE([&] { std::for_each_n(policy, std::data(a), std::size(a), [](int) { throw int{}; }); });
    EXPECT_STD_TERMINATE([&] {
      try {
        (void)std::for_each_n(policy, util::throw_on_move_iterator(std::begin(a), 1), std::size(a), [](int) {});
      } catch (const util::iterator_error&) {
        assert(false);
      }
      std::terminate(); // make the test pass in case the algorithm didn't move the iterator
    });
  });
}
