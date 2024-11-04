//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ranges>

// Call begin() on drop_while_view with empty predicate

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-exceptions
// REQUIRES: libcpp-hardening-mode={{extensive|debug}}
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

#include <ranges>

#include "check_assertion.h"

struct Exception {};
struct ThrowOnCopyPred {
  ThrowOnCopyPred() = default;
  ThrowOnCopyPred(const ThrowOnCopyPred&) { throw Exception{}; }
  ThrowOnCopyPred& operator=(const ThrowOnCopyPred&) = delete;

  ThrowOnCopyPred(ThrowOnCopyPred&&)            = default;
  ThrowOnCopyPred& operator=(ThrowOnCopyPred&&) = default;

  bool operator()(int) const { return false; }
};

int main(int, char**) {
  int input[] = {1, 2, 3};
  auto v1     = std::views::drop_while(input, ThrowOnCopyPred{});
  auto v2     = std::views::drop_while(input, ThrowOnCopyPred{});
  try {
    v1 = v2;
  } catch (...) {
  }
  TEST_LIBCPP_ASSERT_FAILURE(
      v1.begin(),
      "drop_while_view needs to have a non-empty predicate before calling begin() -- did a "
      "previous assignment to this drop_while_view fail?");

  return 0;
}
