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

// check that std::equal(ExecutionPolicy) terminates on user-thrown exceptions

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
    EXPECT_STD_TERMINATE([&] {
      try {
        int a[] = {1, 2};
        int b[] = {1, 2};
        (void)std::equal(policy,
                         util::throw_on_move_iterator(std::begin(a), 1),
                         util::throw_on_move_iterator(std::end(a), 1),
                         util::throw_on_move_iterator(std::begin(b), 1));
      } catch (const util::iterator_error&) {
        assert(false);
      }
      std::terminate(); // make the test pass in case the algorithm didn't move the iterator
    });
    EXPECT_STD_TERMINATE([&] {
      try {
        int a[] = {1, 2};
        int b[] = {1, 2};
        (void)std::equal(
            policy,
            util::throw_on_move_iterator(std::begin(a), 1),
            util::throw_on_move_iterator(std::end(a), 1),
            util::throw_on_move_iterator(std::begin(b), 1),
            util::throw_on_move_iterator(std::end(b), 1));
      } catch (const util::iterator_error&) {
        assert(false);
      }
      std::terminate(); // make the test pass in case the algorithm didn't move the iterator
    });
  });
}
