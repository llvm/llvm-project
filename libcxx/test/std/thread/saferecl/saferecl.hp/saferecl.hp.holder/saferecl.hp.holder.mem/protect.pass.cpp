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

// template<class T> T* protect(const atomic<T*>& src) noexcept;
//   Equivalent to: T* ptr = src.load(relaxed); while (!try_protect(ptr, src)) {} return ptr;

#include <hazard_pointer>
#include <atomic>
#include <cassert>

#include "test_macros.h"

struct Node : std::hazard_pointer_obj_base<Node> {
  int value;
  explicit Node(int v) : value(v) {}
};

int main(int, char**) {
  std::hazard_pointer h = std::make_hazard_pointer();
  {
    Node a(1);
    std::atomic<Node*> src{&a};
    Node* p = h.protect(src);
    assert(p == &a);
    assert(p->value == 1);
    // Protecting again (a new epoch) works and returns the current value.
    Node b(2);
    src.store(&b);
    Node* q = h.protect(src);
    assert(q == &b);
    assert(q->value == 2);
  }
  {
    // A null source: protect returns nullptr and the hazard pointer ends up unassociated.
    std::atomic<Node*> src{nullptr};
    Node* p = h.protect(src);
    assert(p == nullptr);
  }
  {
    // Works with a pointer to a derived-from-Node? No: T must be exactly the hazard-protectable type.
    // (Diagnostics are covered by libcxx/test/libcxx/thread/saferecl/saferecl.hp/hazard_protectable.verify.cpp)
    // Works with two hazard pointers protecting the same object.
    Node a(3);
    std::atomic<Node*> src{&a};
    std::hazard_pointer h2 = std::make_hazard_pointer();
    assert(h.protect(src) == &a);
    assert(h2.protect(src) == &a);
  }
  {
    // src may be a const object, not merely a reference to const.
    Node a(4);
    const std::atomic<Node*> csrc{&a};
    assert(h.protect(csrc) == &a);
  }
  return 0;
}
