//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17
// UNSUPPORTED: no-threads

#include <atomic>

// This compile-only test checks that __atomic_wait, __atomic_notify_one, and __atomic_notify_all are available in C++17

void test_atomic_wait(std::atomic<int>& val) {
  std::__atomic_wait(val, 0, std::memory_order_seq_cst); // expected-no-diagnostics
  std::__atomic_notify_one(val);                         // expected-no-diagnostics
  std::__atomic_notify_all(val);                         // expected-no-diagnostics
}
