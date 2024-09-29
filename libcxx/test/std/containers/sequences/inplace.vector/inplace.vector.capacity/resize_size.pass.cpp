//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <inplace_vector>

// void resize(size_type sz);

#include <inplace_vector>
#include <cassert>

#include "test_macros.h"
#include "MoveOnly.h"

constexpr bool tests() {
  {
    std::inplace_vector<int, 400> v(100);
    v.resize(50);
    assert(v.size() == 50);
    for (int i : v) {
      assert(i == 0);
    }
    v.resize(200);
    assert(v.size() == 200);
    for (int i : v) {
      assert(i == 0);
    }
  }
  if !consteval {
    std::inplace_vector<MoveOnly, 400> v(100);
    v.resize(50);
    assert(v.size() == 50);
    v.resize(200);
    assert(v.size() == 200);
  }

  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());
  return 0;
}
