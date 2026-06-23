//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// <cmath>
//
// [sf.cmath.assoc.laguerre], associated Laguerre polynomials
// float               assoc_laguerref(unsigned n, unsigned m, float x);

#include <cassert>
#include <cmath>

int main() {
  // Single value tested against known solution.
  // Note, underlying Boost.Math is itself well-tested.
  assert(std::abs(std::assoc_laguerref(2, 10, 0.5f) - 60.125f) < 0.001f);
}
