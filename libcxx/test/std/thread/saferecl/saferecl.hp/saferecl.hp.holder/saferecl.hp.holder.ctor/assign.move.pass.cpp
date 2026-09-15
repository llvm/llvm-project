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

// hazard_pointer& operator=(hazard_pointer&& other) noexcept;
//   If this == &other: no effect. Otherwise, if *this is not empty, destroys the hazard pointer owned
//   by *this. Postconditions: *this owns other's hazard pointer (if any); other is empty.

#include <hazard_pointer>
#include <atomic>
#include <cassert>
#include <utility>

#include "test_macros.h"

struct Node : std::hazard_pointer_obj_base<Node> {};

int main(int, char**) {
  {
    // empty = empty
    std::hazard_pointer a, b;
    a = std::move(b);
    assert(a.empty());
    assert(b.empty());
  }
  {
    // nonempty = empty
    std::hazard_pointer a = std::make_hazard_pointer(), b;
    a                     = std::move(b);
    assert(a.empty());
    assert(b.empty());
  }
  {
    // empty = nonempty
    std::hazard_pointer a, b = std::make_hazard_pointer();
    a = std::move(b);
    assert(!a.empty());
    assert(b.empty());
  }
  {
    // nonempty = nonempty
    Node node;
    std::atomic<Node*> src{&node};
    std::hazard_pointer a = std::make_hazard_pointer(), b = std::make_hazard_pointer();
    (void)a.protect(src);
    a = std::move(b);
    assert(!a.empty());
    assert(b.empty());
    assert(a.protect(src) == &node);
  }
  {
    // self-move: no effect
    std::hazard_pointer a    = std::make_hazard_pointer();
    std::hazard_pointer& ref = a;
    a                        = std::move(ref);
    assert(!a.empty());
    std::hazard_pointer e;
    std::hazard_pointer& eref = e;
    e                         = std::move(eref);
    assert(e.empty());
  }
  {
    // returns *this
    std::hazard_pointer a, b = std::make_hazard_pointer();
    std::hazard_pointer& r = (a = std::move(b));
    assert(&r == &a);
  }
  return 0;
}
