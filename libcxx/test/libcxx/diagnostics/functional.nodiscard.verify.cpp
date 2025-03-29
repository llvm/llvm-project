//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// check that <functional> functions are marked [[nodiscard]]

#include <functional>

#include "test_macros.h"

void test() {
  int i = 0;
  std::identity()(i); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
