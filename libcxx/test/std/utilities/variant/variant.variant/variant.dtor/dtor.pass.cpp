//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <variant>

// template <class ...Types> class variant;

// ~variant();

#include <cassert>
#include <type_traits>
#include <variant>

#include "test_macros.h"

struct NonTDtor {
  int* count;
  constexpr NonTDtor(int* a, int*) : count(a) {}
  TEST_CONSTEXPR_CXX20 ~NonTDtor() { ++*count; }
};
static_assert(!std::is_trivially_destructible<NonTDtor>::value, "");

struct NonTDtor1 {
  int* count;
  constexpr NonTDtor1(int*, int* b) : count(b) {}
  TEST_CONSTEXPR_CXX20 ~NonTDtor1() { ++*count; }
};
static_assert(!std::is_trivially_destructible<NonTDtor1>::value, "");

struct TDtor {
  constexpr TDtor() = default;
  constexpr TDtor(const TDtor&) {} // non-trivial copy
  TEST_CONSTEXPR_CXX20 ~TDtor() = default;
};
static_assert(!std::is_trivially_copy_constructible<TDtor>::value, "");
static_assert(std::is_trivially_destructible<TDtor>::value, "");

TEST_CONSTEXPR_CXX20 bool test() {
  {
    using V = std::variant<int, long, TDtor>;
    static_assert(std::is_trivially_destructible<V>::value, "");
    [[maybe_unused]] V v(std::in_place_index<2>);
  }
  {
    using V = std::variant<NonTDtor, int, NonTDtor1>;
    static_assert(!std::is_trivially_destructible<V>::value, "");
    {
      int count0 = 0;
      int count1 = 0;
      {
        V v(std::in_place_index<0>, &count0, &count1);
        assert(count0 == 0);
        assert(count1 == 0);
      }
      assert(count0 == 1);
      assert(count1 == 0);
    }
    {
      int count0 = 0;
      int count1 = 0;
      { V v(std::in_place_index<1>); }
      assert(count0 == 0);
      assert(count1 == 0);
    }
    {
      int count0 = 0;
      int count1 = 0;
      {
        V v(std::in_place_index<2>, &count0, &count1);
        assert(count0 == 0);
        assert(count1 == 0);
      }
      assert(count0 == 0);
      assert(count1 == 1);
    }
  }

  return true;
}

int main(int, char**) {
  test();

#if TEST_STD_VER >= 20
  static_assert(test());
#endif

  return 0;
}
