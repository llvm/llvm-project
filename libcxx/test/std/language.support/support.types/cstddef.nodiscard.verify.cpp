//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// check that <cstddef> functions are marked [[nodiscard]]

#include <cstddef>

#include "test_macros.h"

void test() {
  std::byte b{42};
  std::to_integer<int>(b); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
