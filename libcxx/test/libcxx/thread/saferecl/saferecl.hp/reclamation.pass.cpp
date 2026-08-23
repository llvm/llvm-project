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

// Implementation-specific reclamation behaviour: libc++ runs a reclamation pass inline in retire()
// once max(1000, 2 * <number of hazard pointer records>) objects have been retired since the previous
// pass, so retiring a few thousand objects from a thread with a handful of hazard pointers guarantees
// passes. Numbers below leave generous slack so tuning the constants does not break the test.

#include <hazard_pointer>
#include <atomic>
#include <cassert>
#include <vector>

#include "test_macros.h"

// Namespace-scope counters: a deleter may run well after the block that retired the object has
// exited (the engine leaves up to `threshold` retirements pending at any time), so no deleter may
// write through a pointer to an automatic-storage-duration object.
std::atomic<int> dummies_deleted{0};
struct Dummy : std::hazard_pointer_obj_base<Dummy> {
  ~Dummy() { ++dummies_deleted; }
};

std::atomic<bool> node_deleted{false};
struct Node : std::hazard_pointer_obj_base<Node> {
  ~Node() { node_deleted.store(true); }
};

// Retire enough objects to guarantee at least two passes.
void force_passes() {
  for (int i = 0; i < 5000; ++i)
    (new Dummy)->retire();
}

// A deleter that itself retires another object and uses hazard pointers (re-entrancy).
std::atomic<int> nested_deleted{0};
struct Nested : std::hazard_pointer_obj_base<Nested> {
  int depth;
  explicit Nested(int d) : depth(d) {}
  ~Nested() {
    ++nested_deleted;
    if (depth > 0)
      (new Nested(depth - 1))->retire();
    std::hazard_pointer h = std::make_hazard_pointer(); // acquire/release inside a deleter
    std::atomic<Nested*> src{nullptr};
    (void)h.protect(src);
  }
};

struct Noop {
  void operator()(struct StackObj*) const noexcept {}
};
struct StackObj : std::hazard_pointer_obj_base<StackObj, Noop> {};

int main(int, char**) {
  {
    // Unprotected objects get reclaimed: after 5000 retirements at most ~1000 (+ slack) are still pending.
    int before = dummies_deleted.load();
    force_passes();
    assert(dummies_deleted.load() - before >= 3000);
  }
  {
    // A protected object survives passes and is reclaimed once unprotected and another pass runs.
    node_deleted          = false;
    Node* n               = new Node();
    std::hazard_pointer h = std::make_hazard_pointer();
    h.reset_protection(n);
    n->retire();
    force_passes();
    assert(!node_deleted.load());
    h.reset_protection();
    force_passes();
    assert(node_deleted.load());
  }
  {
    // ~hazard_pointer() ends the epoch as well.
    node_deleted = false;
    Node* n      = new Node();
    {
      std::hazard_pointer h = std::make_hazard_pointer();
      h.reset_protection(n);
      n->retire();
      force_passes();
      assert(!node_deleted.load());
    }
    force_passes();
    assert(node_deleted.load());
  }
  {
    // Move-assignment over a nonempty hazard_pointer ends its epoch.
    node_deleted          = false;
    Node* n               = new Node();
    std::hazard_pointer h = std::make_hazard_pointer();
    h.reset_protection(n);
    n->retire();
    h = std::hazard_pointer(); // releases the associated hazard pointer
    force_passes();
    assert(node_deleted.load());
  }
  {
    // Deleters may retire and use hazard pointers (reclamation is re-entrant).
    (new Nested(3))->retire();
    force_passes();
    force_passes();
    assert(nested_deleted.load() >= 1);
    // Drain: everything retired so far is unprotected, so a couple more passes reclaim the chain.
    force_passes();
    force_passes();
    assert(nested_deleted.load() == 4);
  }
  {
    // Objects with automatic storage and a no-op deleter: retire, force a pass, then they may go away.
    std::vector<StackObj> objs(100);
    for (StackObj& o : objs)
      o.retire();
    force_passes(); // reclaims (no-op deleter) and unlinks them: destroying `objs` is now safe
  }
  return 0;
}
