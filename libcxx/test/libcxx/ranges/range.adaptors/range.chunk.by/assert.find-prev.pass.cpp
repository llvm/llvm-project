//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: no-exceptions
// UNSUPPORTED: !libcpp-hardening-mode=debug
// XFAIL: availability-verbose_abort-missing

// <ranges>

// Call find-prev() on chunk_by_view with begin iterator
// Call find-prev() on chunk_by_view with empty predicate

#include <functional>
#include <ranges>

#include "check_assertion.h"
#include "types.h"

int main(int, char**) {
  int input[] = {1, 1, 2, 2};

  { // Call find-prev() on chunk_by_view with begin iterator
    auto view = std::views::chunk_by(input, std::equal_to{});
    auto it   = view.begin();
    TEST_LIBCPP_ASSERT_FAILURE(--it, "Trying to call __find_prev() on a begin iterator.");
  }

  { // Call find-prev() on chunk_by_view with empty predicate
    auto view1 = std::views::chunk_by(input, ThrowOnCopyPred{});
    auto view2 = std::views::chunk_by(input, ThrowOnCopyPred{});
    auto it    = view1.begin();
    ++it;
    try {
      view1 = view2;
    } catch (...) {
    }
    TEST_LIBCPP_ASSERT_FAILURE(
        --it, "Trying to call __find_prev() on a chunk_by_view that does not have a valid predicate.");
  }

  return 0;
}
