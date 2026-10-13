//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_LIBCXX_ATOMICS_ASYMMETRIC_THREAD_FENCE_STORE_BUFFERING_H
#define TEST_LIBCXX_ATOMICS_ASYMMETRIC_THREAD_FENCE_STORE_BUFFERING_H

#include <__atomic/asymmetric_thread_fence.h>
#include <atomic>
#include <cassert>
#include <thread>

#include "make_test_thread.h"

// A two-thread spin barrier: neither thread leaves arrive_and_wait() before both have entered it.
class SpinBarrier {
public:
  SpinBarrier() : arrived_(0), phase_(0) {}

  void arrive_and_wait() {
    int phase = phase_.load(std::memory_order_acquire);
    if (arrived_.fetch_add(1, std::memory_order_acq_rel) == 1) {
      arrived_.store(0, std::memory_order_relaxed);
      phase_.store(phase + 1, std::memory_order_release);
    } else {
      while (phase_.load(std::memory_order_acquire) == phase)
        std::this_thread::yield();
    }
  }

private:
  std::atomic<int> arrived_;
  std::atomic<int> phase_;
};

using Fence = void (*)(std::memory_order);

inline void light(std::memory_order order) { std::__asymmetric_thread_fence_light(order); }
inline void heavy(std::memory_order order) { std::__asymmetric_thread_fence_heavy(order); }

// Store buffering (Dekker): `x = 1; fence_a; r1 = y` races with `y = 1; fence_b; r2 = x`.
// If the two fences order memory like two seq_cst fences, r1 == 0 && r2 == 0 is impossible.
inline void check_store_buffering(Fence fence_a, Fence fence_b, std::memory_order order, int rounds) {
  std::atomic<int> x(0);
  std::atomic<int> y(0);
  int r2 = -1;
  SpinBarrier barrier;
  std::thread other = support::make_test_thread([&] {
    for (int i = 0; i != rounds; ++i) {
      barrier.arrive_and_wait(); // start of round
      y.store(1, std::memory_order_relaxed);
      fence_b(order);
      r2 = x.load(std::memory_order_relaxed);
      barrier.arrive_and_wait(); // r2 published
      barrier.arrive_and_wait(); // x and y reset
    }
  });
  for (int i = 0; i != rounds; ++i) {
    barrier.arrive_and_wait();
    x.store(1, std::memory_order_relaxed);
    fence_a(order);
    int r1 = y.load(std::memory_order_relaxed);
    barrier.arrive_and_wait();
    assert(!(r1 == 0 && r2 == 0));
    x.store(0, std::memory_order_relaxed);
    y.store(0, std::memory_order_relaxed);
    barrier.arrive_and_wait();
  }
  other.join();
}

#endif // TEST_LIBCXX_ATOMICS_ASYMMETRIC_THREAD_FENCE_STORE_BUFFERING_H
