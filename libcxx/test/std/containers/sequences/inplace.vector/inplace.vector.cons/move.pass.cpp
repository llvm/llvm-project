//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <inplace_vector>

// inplace_vector(inplace_vector&&);

#include <inplace_vector>
#include <cassert>

#include "test_macros.h"
#include "MoveOnly.h"

constexpr bool tests() {
  if !consteval {
    std::inplace_vector<MoveOnly, 10> l;
    std::inplace_vector<MoveOnly, 10> lo;
    for (int i = 1; i <= 3; ++i) {
      l.push_back(i);
      lo.push_back(i);
    }
    std::inplace_vector<MoveOnly, 10> l2 = std::move(l);
    assert(l2.size() == 3);
    assert(l2[0] == 1);
    assert(l2[1] == 2);
    assert(l2[2] == 3);
    assert(l2 == lo);
    // assert(l.size() == lo.size()); // l is left unspecified
  }
  {
    std::inplace_vector<int, 10> l;
    std::inplace_vector<int, 10> lo;
    for (int i = 1; i <= 3; ++i) {
      l.push_back(i);
      lo.push_back(i);
    }
    std::inplace_vector<int, 10> l2 = std::move(l);
    assert(l2 == lo);
    // assert(l.size() == lo.size()); // l is left unspecified
  }

  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());
  return 0;
}
