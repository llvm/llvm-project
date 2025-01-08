//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// check that <vector> functions are marked [[nodiscard]]

#include <vector>

void test_vector() {
  std::vector<int> vector;
  vector.empty(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}

void test_vector_bool() {
  std::vector<bool> vector;
  vector.empty(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
