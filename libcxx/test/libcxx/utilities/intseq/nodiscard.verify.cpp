//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <utility>

// Check that functions are marked [[nodiscard]]

#include <utility>

void test() {
  std::integer_sequence<int, 49, 82, 94> seq;

  seq.size(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
