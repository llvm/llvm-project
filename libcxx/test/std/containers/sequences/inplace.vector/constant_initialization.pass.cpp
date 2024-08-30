//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <inplace_vector>

#include <inplace_vector>
#include <type_traits>
#include <cassert>

struct NonTrivial {
  bool constant = std::is_constant_evaluated();
};

constexpr bool tests() {
  {
    using V = std::inplace_vector<int, 0>;
    if !consteval {
      constinit static V v1;
      assert(v1.size() == 0);
    }
    constexpr static V v2;
    static_assert(v2.size() == 0);
  }
  {
    using V = std::inplace_vector<int, 10>;
    if !consteval {
      constinit static V v1;
      assert(v1.size() == 0);
    }
    constexpr static V v2;
    static_assert(v2.size() == 0);
  }
  {
    using V = std::inplace_vector<int, 10>;
    if !consteval {
      constinit static V v1{1, 2, 3};
      assert(v1.size() == 3);
      assert(v1[0] == 1 && v1[1] == 2 && v1[2] == 3);
    }
    constexpr static V v2{1, 2, 3};
    constexpr static V v3 = [] consteval { return V{1, 2, 3}; }();
    static_assert(v2.size() == 3);
    static_assert(v3.size() == 3);
    static_assert(v2[0] == 1 && v2[1] == 2 && v2[2] == 3);
    static_assert(v3[0] == 1 && v3[1] == 2 && v3[2] == 3);
  }
  {
    using V = std::inplace_vector<NonTrivial, 0>;
    if !consteval {
      constinit static V v1;
      assert(v1.size() == 0);
    }
    constexpr static V v2;
    static_assert(v2.size() == 0);
  }
  if !consteval {
    using V = std::inplace_vector<NonTrivial, 10>;
    static V v(3);
    assert(v.size() == 3);
    assert(!v[0].constant && !v[1].constant && !v[2].constant);
  }

  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());
}
