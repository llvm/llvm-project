//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check that format functions are marked [[clang::lifetimebound]] as a conforming extension
// UNSUPPORTED: c++03, c++11, c++14, c++17
#include <format>
#include "test_macros.h"
int j;
auto test_format() {
  int i = 0;
  return std::make_format_args(
      j, i); // expected-warning {{address of stack memory associated with local variable 'i' returned}}
}
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
auto test_wformat() {
  int i = 0;
  return std::make_wformat_args(
      i, j); // expected-warning {{address of stack memory associated with local variable 'i' returned}}
}
#endif // TEST_HAS_NO_WIDE_CHARACTERS
