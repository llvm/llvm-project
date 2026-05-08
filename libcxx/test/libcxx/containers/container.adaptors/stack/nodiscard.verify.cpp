//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// check that <stack> functions are marked [[nodiscard]]

#include <stack>

void test() {
  std::stack<int> st;
  const std::stack<int> cst;

  st.empty(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  st.size();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  st.top();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cst.top();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
