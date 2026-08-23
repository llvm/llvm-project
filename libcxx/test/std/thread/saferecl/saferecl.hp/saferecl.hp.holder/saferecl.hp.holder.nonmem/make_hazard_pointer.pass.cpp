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

// hazard_pointer make_hazard_pointer();

#include <hazard_pointer>
#include <cassert>
#include <vector>

#include "test_macros.h"

int main(int, char**) {
  {
    std::hazard_pointer h = std::make_hazard_pointer();
    assert(!h.empty());
  }
  {
    // Many nonempty hazard pointers alive at once.
    std::vector<std::hazard_pointer> hps;
    for (int i = 0; i < 100; ++i)
      hps.push_back(std::make_hazard_pointer());
    for (const std::hazard_pointer& h : hps)
      assert(!h.empty());
  }
  {
    // Repeated construction/destruction keeps working (hazard pointers are pooled and reused).
    for (int i = 0; i < 10000; ++i) {
      std::hazard_pointer h = std::make_hazard_pointer();
      assert(!h.empty());
    }
  }
  return 0;
}
