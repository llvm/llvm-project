//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <atomic>
//
// atomic_init

#include <atomic>

void test() {
  std::atomic<int> a;
  std::atomic_init(&a, 1); // expected-warning {{'atomic_init<int>' is deprecated}}

  volatile std::atomic<int> b;
  std::atomic_init(&b, 1); // expected-warning {{'atomic_init<int>' is deprecated}}
}
