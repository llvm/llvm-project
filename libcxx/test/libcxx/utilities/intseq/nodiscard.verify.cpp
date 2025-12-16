//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++14

// <utility>

// Check that functions are marked [[nodiscard]]

#include <utility>

void test() {
  std::integer_sequence<int, 49, 82, 94> seq;

  seq.size(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#if TEST_STD_VER >= 26
  get<0>(seq); // expected-warning {{ignoring return value of function}}
#endif
}
