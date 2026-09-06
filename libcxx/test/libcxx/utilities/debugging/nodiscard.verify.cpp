//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// UNSUPPORTED: availability-debugging-missing

// Check that functions are marked [[nodiscard]]

#include <debugging>

#include "test_macros.h"

void test() {
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::is_debugger_present();
}
