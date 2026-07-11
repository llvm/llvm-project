//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// "support/runway_sample.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <type_traits>

#include "test_macros.h"
#include "runway_sample.h"

constexpr bool test() {
  {
    bool fail = false;
    runway_sample(0, [&](size_t /*i*/) { fail = true; });
    if (fail) {
      return false;
    }
  }
  {
    size_t expected[] = {0};
    size_t n          = 0;
    bool fail         = false;
    runway_sample(0, [&](size_t i) {
      if (expected[n++] != i)
        fail = true;
    });
    if (fail) {
      return false;
    }
  }
  {
    size_t expected[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    size_t n          = 0;
    bool fail         = false;
    runway_sample(10, [&](size_t i) {
      if (expected[n++] != i)
        fail = true;
    });
    if (fail) {
      return false;
    }
  }
  {
    size_t expected[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29};
    size_t n  = 0;
    bool fail = false;
    runway_sample(30, [&](size_t i) {
      if (expected[n++] != i)
        fail = true;
    });
    if (fail) {
      return false;
    }
  }
  {
    size_t expected[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                         50, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99};
    size_t n          = 0;
    bool fail         = false;
    runway_sample(100, [&](size_t i) {
      if (expected[n++] != i)
        fail = true;
    });
    if (fail) {
      return false;
    }
  }
  {
    size_t expected[] = {0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  50,
                         157, 493, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999};
    size_t n          = 0;
    bool fail         = false;
    runway_sample(1000, [&](size_t i) {
      if (expected[n++] != i)
        fail = true;
    });
    if (fail) {
      return false;
    }
  }
  return true;
}

int main() {
  assert(test());
  static_assert(test());
  return 0;
}
