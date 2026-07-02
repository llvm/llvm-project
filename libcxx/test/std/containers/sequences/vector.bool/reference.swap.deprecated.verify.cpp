//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <vector>

// vector<bool>;
// static void swap(reference x, reference y) noexcept; // deprecated in C++26

#include <vector>

#include "test_macros.h"
#include "min_allocator.h"

void test() {
  {
    using VB = std::vector<bool>;
    VB vb(2);
    VB::swap(vb[0], vb[1]); // expected-warning {{'swap' is deprecated}}
  }
  {
    using VB = std::vector<bool, min_allocator<bool>>;
    VB vb(2);
    VB::swap(vb[0], vb[1]); // expected-warning {{'swap' is deprecated}}
  }
  {
    using VB = std::pmr::vector<bool>;
    VB vb(2);
    VB::swap(vb[0], vb[1]); // expected-warning {{'swap' is deprecated}}
  }
}
