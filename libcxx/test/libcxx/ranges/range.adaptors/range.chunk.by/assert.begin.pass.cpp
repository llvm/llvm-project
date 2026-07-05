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
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

// <ranges>

// Call begin() on chunk_by_view with empty predicate

#include <ranges>

#include "check_assertion.h"
#include "types.h"

int main(int, char**) {
  int input[] = {1, 2, 3};
  auto view1  = std::views::chunk_by(input, ThrowOnCopyPred{});
  auto view2  = std::views::chunk_by(input, ThrowOnCopyPred{});
  try {
    view1 = view2;
  } catch (...) {
  }
  TEST_LIBCPP_ASSERT_FAILURE(
      view1.begin(), "Trying to call begin() on a chunk_by_view that does not have a valid predicate.");
  return 0;
}
