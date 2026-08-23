//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// UNSUPPORTED: no-threads
// XFAIL: availability-hazard_pointer-missing

// <hazard_pointer>

// Many threads retiring concurrently, none of them protecting a retired object: reclamation passes race
// with each other and with retire() on the shared retired-object count, and the accounting must stay
// consistent. libc++ runs a pass inline in retire() once max(1000, 2 * <records>) objects are pending, so
// once every thread has finished, the number of objects still pending is bounded by that threshold plus a
// claim in flight per thread, and one more burst of retirements from a single thread drains them all.
// Under ASan/TSan this is also a use-after-free / data-race detector for overlapping passes.

#include <hazard_pointer>
#include <atomic>
#include <cassert>
#include <thread>
#include <vector>

#include "make_test_thread.h"
#include "test_macros.h"

// Namespace scope: deleters run on whichever thread's retire() happens to trigger the pass.
std::atomic<int> deleted{0};
struct Node : std::hazard_pointer_obj_base<Node> {
  ~Node() { ++deleted; }
};

// Never retired: gives every pass some real protections to scan past.
struct Live : std::hazard_pointer_obj_base<Live> {};

int main(int, char**) {
#if defined(TEST_IS_EXECUTED_IN_A_SLOW_ENVIRONMENT)
  const int threads = 4, per_thread = 5000;
#else
  const int threads = 8, per_thread = 50000;
#endif
  constexpr int threshold = 1000; // records stay well below 500, so the threshold is the fixed floor

  std::atomic<bool> go{false};
  std::vector<std::thread> ts;
  for (int t = 0; t < threads; ++t) {
    ts.push_back(support::make_test_thread([&] {
      Live live;
      std::atomic<Live*> src{&live};
      std::hazard_pointer hps[4];
      for (std::hazard_pointer& h : hps) {
        h = std::make_hazard_pointer();
        (void)h.protect(src);
      }
      while (!go.load(std::memory_order_acquire)) {
      }
      for (int i = 0; i < per_thread; ++i)
        (new Node)->retire();
    }));
  }
  go.store(true, std::memory_order_release);
  for (std::thread& t : ts)
    t.join();

  const int retired = threads * per_thread;
  const int pending = retired - deleted.load();
  assert(pending >= 0);                         // nothing reclaimed twice
  assert(pending <= threshold * (threads + 1)); // bounded: no credit leaked out of the counter

  // Everything retired above is unprotected; the first pass triggered here extracts and reclaims it all.
  for (int i = 0; i < 3 * threshold; ++i)
    (new Node)->retire();
  assert(deleted.load() >= retired);
  return 0;
}
