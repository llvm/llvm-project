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

// hazard_pointer_obj_base's protected special members are defaulted. A copy or move of an object --
// even of one that has already been retired but is still protected -- is a fresh object that can itself
// be retired; assignment never affects retirement state.

#include <hazard_pointer>
#include <atomic>
#include <cassert>
#include <utility>

#include "test_macros.h"

#if defined(TEST_IS_EXECUTED_IN_A_SLOW_ENVIRONMENT)
constexpr int N = 5000;
#else
constexpr int N = 100000; // comfortably above any sane reclamation bound
#endif

std::atomic<int> deleted{0};

struct Node : std::hazard_pointer_obj_base<Node> {
  int value;
  explicit Node(int v) : value(v) {}
  Node(const Node&)            = default;
  Node(Node&&)                 = default;
  Node& operator=(const Node&) = default;
  Node& operator=(Node&&)      = default;
  ~Node() { ++deleted; }
};

void force_reclamation() {
  struct Dummy : std::hazard_pointer_obj_base<Dummy> {};
  for (int i = 0; i < N; ++i)
    (new Dummy)->retire();
}

int main(int, char**) {
  {
    // Copy of a live object, then retire both.
    Node* a = new Node(1);
    Node* b = new Node(*a);
    assert(b->value == 1);
    a->retire();
    b->retire();
  }
  {
    // Copy and move of a retired-but-protected object.
    Node* a               = new Node(2);
    std::hazard_pointer h = std::make_hazard_pointer();
    h.reset_protection(a);
    a->retire();
    Node* copy  = new Node(*a);            // fresh, unretired
    Node* moved = new Node(std::move(*a)); // fresh, unretired
    assert(copy->value == 2 && moved->value == 2);
    copy->retire();
    moved->retire();
    h.reset_protection();
  }
  {
    // Assignment from a retired object into a live one leaves the target retirable.
    Node* a               = new Node(3);
    Node* b               = new Node(4);
    std::hazard_pointer h = std::make_hazard_pointer();
    h.reset_protection(a);
    a->retire();
    *b = *a;
    assert(b->value == 3);
    b->retire();
    Node* c = new Node(5);
    *c      = std::move(*a);
    assert(c->value == 3);
    c->retire();
    h.reset_protection();
  }
  int before = deleted.load();
  force_reclamation();
  assert(deleted.load() > before); // some of the objects above were reclaimed exactly once each; the
                                   // total is checked by ASan (no double free) rather than by count.
  return 0;
}
