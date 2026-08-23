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

// hazard_pointer(hazard_pointer&& other) noexcept;
//   Postconditions: If other is empty, *this is empty. Otherwise *this owns the hazard pointer
//   originally owned by other; other is empty.

#include <hazard_pointer>
#include <atomic>
#include <cassert>
#include <utility>

#include "test_macros.h"

struct Node : std::hazard_pointer_obj_base<Node> {
  int value = 0;
};

int main(int, char**) {
  {
    std::hazard_pointer empty;
    std::hazard_pointer h(std::move(empty));
    assert(h.empty());
    assert(empty.empty());
  }
  {
    Node node;
    std::atomic<Node*> src{&node};
    std::hazard_pointer a = std::make_hazard_pointer();
    Node* p               = a.protect(src);
    assert(p == &node);
    std::hazard_pointer b(std::move(a));
    assert(!b.empty());
    assert(a.empty());
    // b owns the hazard pointer, still associated with node; b can keep using it.
    b.reset_protection();
    Node* q = b.protect(src);
    assert(q == &node);
  }
  return 0;
}
