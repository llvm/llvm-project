//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// REQUIRES: has-unix-headers
// REQUIRES: libcpp-hardening-mode=debug
// XFAIL: availability-verbose_abort-missing

#include "test_macros.h"

TEST_DIAGNOSTIC_PUSH
TEST_CLANG_DIAGNOSTIC_IGNORED("-Wprivate-header")
#include <__utility/pointer_int_pair.h>
TEST_DIAGNOSTIC_POP

#include <cassert>

#include "check_assertion.h"

struct [[gnu::packed]] Packed {
  char c;
  int i;
};

int main(int, char**) {
  TEST_LIBCPP_ASSERT_FAILURE(
      (std::__pointer_int_pair<int*, size_t, std::__integer_width{1}>{nullptr, 2}), "integer is too large!");

  TEST_DIAGNOSTIC_PUSH
  TEST_CLANG_DIAGNOSTIC_IGNORED("-Waddress-of-packed-member") // That's what we're trying to test
  TEST_GCC_DIAGNOSTIC_IGNORED("-Waddress-of-packed-member")
  alignas(int) Packed p;
  TEST_LIBCPP_ASSERT_FAILURE(
      (std::__pointer_int_pair<int*, size_t, std::__integer_width{1}>{&p.i, 0}), "Pointer alignment is too low!");
  TEST_DIAGNOSTIC_POP

  return 0;
}
