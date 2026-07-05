//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <valarray>

// class slice;

// friend bool operator==(const slice& x, const slice& y);

#include <cassert>
#include <valarray>

#include "test_comparisons.h"

void test() {
  {
    std::slice s1;
    std::slice s2;

    assert(testEquality(s1, s2, true));
  }
  {
    std::slice s1{1, 2, 3};
    std::slice s2{1, 2, 3};

    assert(testEquality(s1, s2, true));
  }
  {
    std::slice s1;
    std::slice s2{1, 2, 3};

    assert(testEquality(s1, s2, false));
  }
  {
    std::slice s1{0, 2, 3};
    std::slice s2{1, 2, 3};

    assert(testEquality(s1, s2, false));
  }
  {
    std::slice s1{1, 0, 3};
    std::slice s2{1, 2, 3};

    assert(testEquality(s1, s2, false));
  }
  {
    std::slice s1{1, 2, 0};
    std::slice s2{1, 2, 3};

    assert(testEquality(s1, s2, false));
  }
}

int main(int, char**) {
  test();

  return 0;
}
