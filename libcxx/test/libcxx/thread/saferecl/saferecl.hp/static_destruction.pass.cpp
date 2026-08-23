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

// hazard_pointer objects with static storage duration are destroyed during program termination, and
// objects may be retired from destructors of static objects: the domain must outlive them (it is never
// destroyed).

#include <hazard_pointer>
#include <atomic>
#include <cassert>

#include "test_macros.h"

struct Node : std::hazard_pointer_obj_base<Node> {};

std::hazard_pointer global_hp; // released during static destruction
struct RetiresAtExit {
  Node* node = new Node;
  ~RetiresAtExit() {
    node->retire();                                        // retire during static destruction
    std::hazard_pointer late = std::make_hazard_pointer(); // and acquire/release
    assert(!late.empty());
  }
} retires_at_exit;

int main(int, char**) {
  global_hp = std::make_hazard_pointer();
  Node n;
  global_hp.reset_protection(&n);
  global_hp.reset_protection();
  return 0;
}
