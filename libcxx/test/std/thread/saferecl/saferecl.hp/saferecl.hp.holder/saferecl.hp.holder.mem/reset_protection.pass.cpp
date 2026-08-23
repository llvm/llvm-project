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

// template<class T> void reset_protection(const T* ptr) noexcept;
// void reset_protection(nullptr_t = nullptr) noexcept;

#include <hazard_pointer>
#include <atomic>
#include <cassert>

#include "test_macros.h"

struct Node : std::hazard_pointer_obj_base<Node> {};

int main(int, char**) {
  Node a, b;
  std::hazard_pointer h = std::make_hazard_pointer();
  h.reset_protection();        // unassociated -> unassociated
  h.reset_protection(nullptr); // same
  h.reset_protection(&a);      // associate with a
  h.reset_protection(&b);      // re-associate
  const Node* cb = &b;
  h.reset_protection(cb); // pointer-to-const is fine: T is deduced as Node
  Node* null = nullptr;
  h.reset_protection(null); // a null T*: equivalent to reset_protection()
  h.reset_protection();
  // Still usable afterwards.
  std::atomic<Node*> src{&a};
  assert(h.protect(src) == &a);
  return 0;
}
