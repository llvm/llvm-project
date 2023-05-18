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

// check that std::reduce(ExecutionPolicy) terminates on user-thrown exceptions

#include <numeric>

#include "check_assertion.h"
#include "test_execution_policies.h"
#include "test_iterators.h"

int main(int, char**) {
  test_execution_policies([&](auto&& policy) {
    EXPECT_STD_TERMINATE([&] {
      try {
        int a[] = {1, 2};
        (void)std::reduce(
            policy, util::throw_on_move_iterator(std::begin(a), 1), util::throw_on_move_iterator(std::end(a), 1));
      } catch (const util::iterator_error&) {
        assert(false);
      }
      std::terminate(); // make the test pass in case the algorithm didn't move the iterator
    });

    EXPECT_STD_TERMINATE([&] {
      int a[2]{};
      (void)std::reduce(policy, std::begin(a), std::end(a), 1, [](int, int) -> int { throw 1; });
    });
    EXPECT_STD_TERMINATE([&] {
      try {
        int a[] = {1, 2};
        (void)std::reduce(
            policy, util::throw_on_move_iterator(std::begin(a), 1), util::throw_on_move_iterator(std::end(a), 1), 1);
      } catch (const util::iterator_error&) {
        assert(false);
      }
      std::terminate(); // make the test pass in case the algorithm didn't move the iterator
    });
  });
}
