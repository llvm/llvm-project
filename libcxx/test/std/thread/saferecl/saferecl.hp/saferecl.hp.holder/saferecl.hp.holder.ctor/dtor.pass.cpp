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

// ~hazard_pointer();
//   If *this is not empty, destroys the hazard pointer owned by *this, thereby ending its current
//   protection epoch. (Reclaimability after the epoch ends is tested in ../../protection.pass.cpp.)

#include <hazard_pointer>
#include <atomic>
#include <cassert>
#include <utility>

#include "test_macros.h"

struct Node : std::hazard_pointer_obj_base<Node> {};

int main(int, char**) {
  {
    std::hazard_pointer empty; // destroying an empty hazard_pointer is a no-op
  }
  {
    Node node;
    std::atomic<Node*> src{&node};
    std::hazard_pointer h = std::make_hazard_pointer();
    (void)h.protect(src); // destroyed while associated
  }
  {
    std::hazard_pointer a = std::make_hazard_pointer();
    std::hazard_pointer b(std::move(a)); // a is empty: only b destroys the hazard pointer
  }
  // Destroy many, then acquire many again: the pool must be reusable.
  for (int round = 0; round < 3; ++round) {
    std::hazard_pointer hps[64];
    for (std::hazard_pointer& h : hps)
      h = std::make_hazard_pointer();
    for (std::hazard_pointer& h : hps)
      assert(!h.empty());
  }
  return 0;
}
