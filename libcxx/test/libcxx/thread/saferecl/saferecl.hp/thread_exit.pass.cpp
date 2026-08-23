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

// The per-thread record cache is returned to the domain when a thread exits. A hazard_pointer owned by a
// thread_local is destroyed during thread teardown, before or after that cache is gone depending on the
// platform's ordering of thread_local destructors vs. TSD key destructors (on glibc: before); both orders
// must be crash-free. The "cache already gone" path is exercised deterministically by
// thread_exit_late_release.pass.cpp. Also exercises many threads acquiring/releasing to churn the caches
// and the available list. Nothing here is timing dependent; ASan/TSan make it a real detector.

#include <hazard_pointer>
#include <atomic>
#include <cassert>
#include <thread>
#include <vector>

#include "make_test_thread.h"
#include "test_macros.h"

struct Node : std::hazard_pointer_obj_base<Node> {};

// A thread_local whose destructor runs during thread teardown and destroys a hazard_pointer then.
struct LateHolder {
  std::hazard_pointer hp;
  ~LateHolder() {
    // Acquire another one during teardown too: acquire must also work when the cache is dead/absent.
    std::hazard_pointer late = std::make_hazard_pointer();
    assert(!late.empty());
    Node n;
    late.reset_protection(&n);
    late.reset_protection();
  }
};

void thread_body(int rounds) {
  static thread_local LateHolder late;
  late.hp = std::make_hazard_pointer(); // hazard_pointer released after the cache's own destructor
  for (int i = 0; i < rounds; ++i) {
    std::hazard_pointer hps[20]; // more than the cache capacity (9): overflow into the domain
    for (std::hazard_pointer& h : hps)
      h = std::make_hazard_pointer();
    for (std::hazard_pointer& h : hps)
      assert(!h.empty());
  }
}

int main(int, char**) {
#if defined(TEST_IS_EXECUTED_IN_A_SLOW_ENVIRONMENT)
  const int threads = 8, rounds = 200;
#else
  const int threads = 32, rounds = 2000;
#endif
  for (int wave = 0; wave < 3; ++wave) {
    std::vector<std::thread> ts;
    for (int i = 0; i < threads; ++i)
      ts.push_back(support::make_test_thread(thread_body, rounds));
    for (std::thread& t : ts)
      t.join();
  }
  // Records left over from exited threads are reusable by the main thread.
  std::vector<std::hazard_pointer> hps;
  for (int i = 0; i < 1000; ++i)
    hps.push_back(std::make_hazard_pointer());
  return 0;
}
