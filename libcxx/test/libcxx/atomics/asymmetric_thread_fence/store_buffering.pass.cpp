//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++11
// UNSUPPORTED: no-threads

// A light/heavy (or heavy/heavy) pair of asymmetric fences orders memory like a pair of seq_cst fences.
// Two light fences give no guarantee and are not tested.

#include <__atomic/asymmetric_thread_fence.h>
#include <atomic>
#include <thread>

#include "make_test_thread.h"
#include "store_buffering.h"

// Fences are issued from destructors during static destruction and thread exit (where e.g. a hazard
// pointer domain is torn down); they must neither crash nor deadlock there.
struct FenceInDestructor {
  ~FenceInDestructor() {
    std::__asymmetric_thread_fence_light(std::memory_order_seq_cst);
    std::__asymmetric_thread_fence_heavy(std::memory_order_seq_cst);
  }
};

FenceInDestructor fence_at_exit;

int main(int, char**) {
  check_store_buffering(light, heavy, std::memory_order_seq_cst, 100000);
  check_store_buffering(heavy, light, std::memory_order_seq_cst, 100000);
  check_store_buffering(heavy, heavy, std::memory_order_seq_cst, 20000);

  // libc++ ignores the order and always fences as seq_cst (P1202 would allow weaker orders to do less).
  const std::memory_order orders[] = {
      std::memory_order_relaxed,
      std::memory_order_consume,
      std::memory_order_acquire,
      std::memory_order_release,
      std::memory_order_acq_rel,
      std::memory_order_seq_cst};
  for (std::memory_order order : orders)
    check_store_buffering(light, heavy, order, 5000);

  support::make_test_thread([] {
    thread_local FenceInDestructor fence_at_thread_exit;
    (void)&fence_at_thread_exit;
  }).join();

  return 0;
}
