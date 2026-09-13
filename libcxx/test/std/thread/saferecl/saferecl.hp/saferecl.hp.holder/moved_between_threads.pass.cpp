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

// A hazard pointer is owned by an object, not by the thread that made it: a nonempty hazard_pointer may
// be moved to another thread and used and destroyed there. The protection epoch is unaffected by the
// move, so an object retired while the moved-to hazard pointer protects it is not reclaimed.

#include <hazard_pointer>
#include <atomic>
#include <cassert>
#include <thread>
#include <utility>

#include "make_test_thread.h"
#include "test_macros.h"

#if defined(TEST_IS_EXECUTED_IN_A_SLOW_ENVIRONMENT)
constexpr int N = 5000;
#else
constexpr int N = 100000;
#endif

// Namespace scope: a deleter may run long after the block that retired the object has exited.
std::atomic<bool> node_deleted{false};

struct Node : std::hazard_pointer_obj_base<Node> {
  int value = 42;
  ~Node() { node_deleted.store(true); }
};

struct Spare : std::hazard_pointer_obj_base<Spare> {
  int value = 7;
};

struct Dummy : std::hazard_pointer_obj_base<Dummy> {};

void retire_dummies() {
  for (int i = 0; i < N; ++i)
    (new Dummy)->retire();
}

int main(int, char**) {
  Node* n = new Node;
  std::atomic<Node*> src{n};

  Spare spare;
  std::atomic<Spare*> spare_src{&spare};

  std::hazard_pointer h = std::make_hazard_pointer();
  assert(h.protect(src) == n);
  src.store(nullptr);
  n->retire(); // retired while protected by a hazard pointer that is about to change threads

  std::thread t = support::make_test_thread([moved = std::move(h), &spare_src, &spare, n]() mutable {
    // The epoch survived the move: n is still protected here, on a thread that never acquired it.
    assert(!moved.empty());
    retire_dummies();
    assert(!node_deleted.load());
    assert(n->value == 42);

    // The moved-to hazard pointer is fully usable on this thread.
    assert(moved.protect(spare_src) == &spare); // ends n's epoch and starts one for spare
    assert(spare.value == 7);
    moved.reset_protection();
  }); // ~hazard_pointer runs here: the record is released by a thread that did not acquire it
  t.join();

  assert(h.empty()); // moved from
  retire_dummies();  // n is unprotected now; whether it has been reclaimed is unspecified
  return 0;
}
