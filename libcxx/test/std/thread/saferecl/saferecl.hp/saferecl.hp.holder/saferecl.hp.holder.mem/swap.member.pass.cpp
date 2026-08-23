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

// void swap(hazard_pointer& other) noexcept;
//   Swaps ownership; the hazard pointers themselves stay associated with what they protected.

#include <hazard_pointer>
#include <atomic>
#include <cassert>

#include "test_macros.h"

struct Node : std::hazard_pointer_obj_base<Node> {};

template <class T>
concept IsNoThrowMemberSwappable = requires(T& t) {
  { t.swap(t) } noexcept;
};
static_assert(IsNoThrowMemberSwappable<std::hazard_pointer>);

int main(int, char**) {
  {
    std::hazard_pointer a, b;
    a.swap(b);
    assert(a.empty() && b.empty());
  }
  {
    std::hazard_pointer a = std::make_hazard_pointer(), b;
    a.swap(b);
    assert(a.empty());
    assert(!b.empty());
    b.swap(a);
    assert(!a.empty());
    assert(b.empty());
  }
  {
    Node node;
    std::atomic<Node*> src{&node};
    std::hazard_pointer a = std::make_hazard_pointer(), b = std::make_hazard_pointer();
    (void)a.protect(src);
    a.swap(b);
    assert(!a.empty() && !b.empty());
    // No epoch ended or started: b now owns the pointer associated with node and may keep using it.
    assert(b.protect(src) == &node);
    a.swap(a); // self-swap: no effect
    assert(!a.empty());
  }
  return 0;
}
