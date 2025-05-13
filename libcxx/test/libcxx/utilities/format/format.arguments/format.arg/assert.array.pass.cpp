//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <format>

// Formatting non-null-terminated character arrays.

// REQUIRES: std-at-least-c++20, has-unix-headers, libcpp-hardening-mode={{extensive|debug}}
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

#include <format>

#include "check_assertion.h"

int main(int, char**) {
  {
    const char non_null_terminated[3]{'1', '2', '3'};
    TEST_LIBCPP_ASSERT_FAILURE(std::format("{}", non_null_terminated), "formatting a non-null-terminated array");
  }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  {
    const wchar_t non_null_terminated[3]{L'1', L'2', L'3'};
    TEST_LIBCPP_ASSERT_FAILURE(std::format(L"{}", non_null_terminated), "formatting a non-null-terminated array");
  }
#endif

  return 0;
}
