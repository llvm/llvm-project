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

// bool empty() const noexcept;

#include <inplace_vector>
#include <cassert>

#include "test_macros.h"
#include "MoveOnly.h"

constexpr bool tests() {
  {
    using V = std::inplace_vector<int, 10>;
    V v;
    ASSERT_NOEXCEPT(v.empty());
    assert(v.empty());
    v.push_back(1);
    assert(!v.empty());
    v.clear();
    assert(v.empty());
  }
  {
    using V = std::inplace_vector<int, 0>;
    V v;
    ASSERT_NOEXCEPT(v.empty());
    assert(v.empty());
    // Without language extension, since V is an empty type, there is no way for this to *not* be a constant expression
    static_assert(v.empty());
  }
  {
    using V = std::inplace_vector<MoveOnly, 0>;
    V v;
    ASSERT_NOEXCEPT(v.empty());
    assert(v.empty());
    // See above
    static_assert(v.empty());
  }
  if !consteval {
    using V = std::inplace_vector<MoveOnly, 10>;
    V v;
    ASSERT_NOEXCEPT(v.empty());
    assert(v.empty());
    v.push_back(V::value_type(1));
    assert(!v.empty());
    v.clear();
    assert(v.empty());
  }

  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());
  return 0;
}
