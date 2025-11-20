//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// check that <string_view> functions are marked [[nodiscard]]

#include <string_view>

#include "test_macros.h"

void test() {
  std::string_view string_view;
  string_view.empty(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#if TEST_STD_VER >= 26
  string_view.subview(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif
}
