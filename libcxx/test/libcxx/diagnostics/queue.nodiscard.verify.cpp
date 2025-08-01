//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// check that <queue> functions are marked [[nodiscard]]

#include <queue>

void test_queue() {
  std::queue<int> queue;
  queue.empty(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}

void test_priority_queue() {
  std::priority_queue<int> priority_queue;
  priority_queue.empty(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
