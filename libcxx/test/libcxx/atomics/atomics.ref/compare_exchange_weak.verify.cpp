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

// bool compare_exchange_weak(T& expected, T desired, memory_order success,
// memory_order failure) const noexcept;
//
// Preconditions: failure is memory_order::relaxed, memory_order::consume,
// memory_order::acquire, or memory_order::seq_cst.

#include <atomic>

void test() {
  using T = int;

  T x(T(42));
  std::atomic_ref const a(x);

  T expected(T(2));
  T const desired(T(3));
  std::memory_order const success = std::memory_order_relaxed;
  // clang-format off
  a.compare_exchange_weak(expected, desired, success, std::memory_order_relaxed);
  a.compare_exchange_weak(expected, desired, success, std::memory_order_consume);
  a.compare_exchange_weak(expected, desired, success, std::memory_order_acquire);
  a.compare_exchange_weak(expected, desired, success, std::memory_order_seq_cst);
  a.compare_exchange_weak(expected, desired, success, std::memory_order_release); // expected-warning {{memory order argument to atomic operation is invalid}}
  a.compare_exchange_weak(expected, desired, success, std::memory_order_acq_rel); // expected-warning {{memory order argument to atomic operation is invalid}}
  // clang-format on
}
