//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// Check that functions are marked [[nodiscard]]

#include <any>
#include <utility>
#include <vector>

#include "test_macros.h"

void test() {
  std::any a{94};

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  a.has_value();
#if !defined(TEST_HAS_NO_RTTI)
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  a.type();
#endif

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::make_any<int>(82);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::make_any<std::vector<int>>({94, 82, 50});

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::any_cast<const int&>(std::as_const(a));
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::any_cast<int&>(a);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::any_cast<int&&>(std::move(a));

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::any_cast<int*>(&a);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::any_cast<const int*>(&std::as_const(a));
}
