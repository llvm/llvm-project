//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: diagnose-if-support
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <atomic>

// void store(T desired, memory_order order = memory_order::seq_cst) const noexcept;
//
// Preconditions: order is memory_order::relaxed, memory_order::release, or memory_order::seq_cst.

#include <atomic>

void test() {
  using T = int;

  T x(T(1));
  std::atomic_ref const a(x);

  T const desired(T(2));

  // clang-format off
  a.store(desired, std::memory_order_relaxed);
  a.store(desired, std::memory_order_release);
  a.store(desired, std::memory_order_seq_cst);
  a.store(desired, std::memory_order_consume); // expected-warning {{memory order argument to atomic operation is invalid}}
  a.store(desired, std::memory_order_acquire); // expected-warning {{memory order argument to atomic operation is invalid}}
  a.store(desired, std::memory_order_acq_rel); // expected-warning {{memory order argument to atomic operation is invalid}}
  // clang-format on
}
