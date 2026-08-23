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

// template<class T> bool try_protect(T*& ptr, const atomic<T*>& src) noexcept;
//   old = ptr; reset_protection(old); ptr = src.load(acquire); if (old != ptr) reset_protection();
//   Returns old == ptr.

#include <hazard_pointer>
#include <atomic>
#include <cassert>

#include "test_macros.h"

struct Node : std::hazard_pointer_obj_base<Node> {};

int main(int, char**) {
  Node a, b;
  std::hazard_pointer h = std::make_hazard_pointer();
  {
    // src unchanged: succeeds, ptr unchanged.
    std::atomic<Node*> src{&a};
    Node* ptr = src.load();
    assert(h.try_protect(ptr, src));
    assert(ptr == &a);
  }
  {
    // src changed between the caller's load and try_protect: fails and reports the new value.
    std::atomic<Node*> src{&a};
    Node* ptr = src.load();
    src.store(&b);
    assert(!h.try_protect(ptr, src));
    assert(ptr == &b);
    // The reported value is now current: the retry succeeds.
    assert(h.try_protect(ptr, src));
    assert(ptr == &b);
  }
  {
    // A stale null pointer.
    std::atomic<Node*> src{&a};
    Node* ptr = nullptr;
    assert(!h.try_protect(ptr, src));
    assert(ptr == &a);
    assert(h.try_protect(ptr, src));
  }
  {
    // A stale non-null pointer against a now-null source.
    std::atomic<Node*> src{nullptr};
    Node* ptr = &a;
    assert(!h.try_protect(ptr, src));
    assert(ptr == nullptr);
    assert(h.try_protect(ptr, src));
    assert(ptr == nullptr);
  }
  return 0;
}
