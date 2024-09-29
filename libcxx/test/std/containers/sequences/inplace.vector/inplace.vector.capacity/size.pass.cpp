//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <inplace_vector>

// class inplace_vector

// size_type size() const noexcept;

#include <inplace_vector>
#include <cassert>

#include "test_macros.h"
#include "MoveOnly.h"

constexpr bool tests() {
  {
    using V = std::inplace_vector<int, 10>;
    V v;
    ASSERT_NOEXCEPT(v.size());
    assert(v.size() == 0);
    v.push_back(2);
    assert(v.size() == 1);
    v.push_back(1);
    assert(v.size() == 2);
    v.push_back(3);
    assert(v.size() == 3);
    v.erase(v.begin());
    assert(v.size() == 2);
    v.erase(v.begin());
    assert(v.size() == 1);
    v.erase(v.begin());
    assert(v.size() == 0);
  }
  if !consteval {
    using V = std::inplace_vector<MoveOnly, 10>;
    V v;
    ASSERT_NOEXCEPT(v.size());
    assert(v.size() == 0);
    v.push_back(2);
    assert(v.size() == 1);
    v.push_back(1);
    assert(v.size() == 2);
    v.push_back(3);
    assert(v.size() == 3);
    v.erase(v.begin());
    assert(v.size() == 2);
    v.erase(v.begin());
    assert(v.size() == 1);
    v.erase(v.begin());
    assert(v.size() == 0);
  }
  {
    using V = std::inplace_vector<int, 0>;
    V v;
    ASSERT_NOEXCEPT(v.size());
    assert(v.size() == 0);
    // Without language extension, since V is an empty type, there is no way for this to *not* be a constant expression
    static_assert(v.size() == 0);
  }
  {
    using V = std::inplace_vector<MoveOnly, 0>;
    V v;
    ASSERT_NOEXCEPT(v.size());
    assert(v.size() == 0);
    // See above
    static_assert(v.size() == 0);
  }

  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());
  return 0;
}
