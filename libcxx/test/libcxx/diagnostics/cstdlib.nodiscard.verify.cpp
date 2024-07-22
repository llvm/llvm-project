//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// We don't control the implementation of the stdlib.h functions on windows
// UNSUPPORTED: windows

// check that <cstdlib> functions are marked [[nodiscard]]

#include <cmath>
#include "test_macros.h"

void test() {
  std::abs(0l);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::abs(0ll); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::abs(0.f); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::abs(0.);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::abs(0.l); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
