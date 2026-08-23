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

// [saferecl.hp.general]: an object retired while some hazard pointer is associated with it is not
// reclaimed until that protection epoch ends; the number of possibly-reclaimable objects is bounded.

#include <hazard_pointer>
#include <atomic>
#include <cassert>

#include "test_macros.h"

// Namespace-scope counters: a deleter may run well after the block that retired the object has
// exited (the engine leaves up to `threshold` retirements pending at any time), so no deleter may
// write through a pointer to an automatic-storage-duration object.
std::atomic<int> dummies_deleted{0};
struct Dummy : std::hazard_pointer_obj_base<Dummy> {
  ~Dummy() { ++dummies_deleted; }
};

// One flag per block, and each Node reports to the flag it was handed: an object retired by an earlier
// block may still be reclaimed during a later one (when that happens is unspecified), so a single shared
// flag would make one block's assertions depend on another block's reclamation timing.
std::atomic<bool> node_deleted_1{false};
std::atomic<bool> node_deleted_2{false};
std::atomic<bool> node_deleted_3{false};

struct Node : std::hazard_pointer_obj_base<Node> {
  explicit Node(std::atomic<bool>& deleted) : deleted_(&deleted) {}
  std::atomic<bool>* deleted_;
  int value = 42;
  ~Node() { deleted_->store(true); }
};

#if defined(TEST_IS_EXECUTED_IN_A_SLOW_ENVIRONMENT)
constexpr int N = 5000;
#else
constexpr int N = 100000;
#endif

void retire_dummies() {
  for (int i = 0; i < N; ++i)
    (new Dummy)->retire();
}

int main(int, char**) {
  {
    // Protected via protect(): survives reclamation passes; readable throughout.
    int before = dummies_deleted.load();
    Node* n    = new Node(node_deleted_1);
    std::atomic<Node*> src{n};
    std::hazard_pointer h = std::make_hazard_pointer();
    Node* p               = h.protect(src);
    assert(p == n);
    src.store(nullptr);
    n->retire();
    retire_dummies();
    assert(dummies_deleted.load() - before > 0); // reclamation did happen ...
    assert(!node_deleted_1.load());              // ... but not of the protected object
    assert(p->value == 42);
    h.reset_protection(); // epoch ends; n becomes possibly-reclaimable
    retire_dummies();
    // Whether n has been reclaimed by now is unspecified (see libcxx/.../reclamation.pass.cpp).
  }
  {
    // Protected via reset_protection(ptr) then released by ~hazard_pointer(): same guarantee.
    Node* n = new Node(node_deleted_2);
    {
      std::hazard_pointer h = std::make_hazard_pointer();
      h.reset_protection(n);
      n->retire();
      retire_dummies();
      assert(!node_deleted_2.load());
      assert(n->value == 42);
    } // h destroyed: epoch ends
    retire_dummies();
  }
  {
    // Two hazard pointers on the same object: it stays protected until both epochs end.
    Node* n                = new Node(node_deleted_3);
    std::hazard_pointer h1 = std::make_hazard_pointer(), h2 = std::make_hazard_pointer();
    h1.reset_protection(n);
    h2.reset_protection(n);
    n->retire();
    h1.reset_protection();
    retire_dummies();
    assert(!node_deleted_3.load()); // still protected by h2
    h2.reset_protection();
    retire_dummies();
  }
  return 0;
}
