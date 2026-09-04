//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// check that <queue> functions are marked [[nodiscard]]

#include <queue>

void test() {
  {
    std::queue<int> q;
    const std::queue<int> cq;

    q.empty();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    q.size();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    q.front();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    cq.front(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    q.back();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    cq.back();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }

  {
    std::priority_queue<int> pq;

    pq.empty(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    pq.size();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    pq.top();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
}
