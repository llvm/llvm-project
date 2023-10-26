//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: no-exceptions

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// check that std::all_of(ExecutionPolicy) terminates on user-thrown exceptions

#include <algorithm>

#include "check_assertion.h"
#include "test_execution_policies.h"
#include "test_iterators.h"

int main(int, char**) {
  test_execution_policies([](auto&& policy) {
    EXPECT_STD_TERMINATE([&] {
      int a[] = {1, 2};
      (void)std::all_of(policy, std::begin(a), std::end(a), [](int i) -> bool { throw i; });
    });
    EXPECT_STD_TERMINATE([&] {
      try {
        int a[] = {1, 2};
        (void)std::all_of(
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
