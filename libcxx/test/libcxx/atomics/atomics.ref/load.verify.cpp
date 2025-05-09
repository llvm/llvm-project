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

// T load(memory_order order = memory_order::seq_cst) const noexcept;
//
// Preconditions: order is memory_order::relaxed, memory_order::consume, memory_order::acquire, or memory_order::seq_cst.

#include <atomic>

void test() {
  using T = int;

  T x(T(1));
  std::atomic_ref const a(x);

  // clang-format off
  (void)a.load(std::memory_order_relaxed);
  (void)a.load(std::memory_order_consume);
  (void)a.load(std::memory_order_acquire);
  (void)a.load(std::memory_order_seq_cst);
  (void)a.load(std::memory_order_release); // expected-warning {{memory order argument to atomic operation is invalid}}
  (void)a.load(std::memory_order_acq_rel); // expected-warning {{memory order argument to atomic operation is invalid}}
  // clang-format on
}
