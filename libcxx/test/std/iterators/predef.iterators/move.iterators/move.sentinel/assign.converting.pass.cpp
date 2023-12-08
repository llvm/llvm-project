//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <iterator>

// move_sentinel

// template<class S2>
//   requires assignable_from<S&, const S2&>
//     constexpr move_sentinel& operator=(const move_sentinel<S2>& s);

#include <cassert>
#include <concepts>
#include <iterator>
#include <type_traits>

struct NonAssignable {
  NonAssignable& operator=(int i);
};
static_assert(std::semiregular<NonAssignable>);
static_assert(std::is_assignable_v<NonAssignable, int>);
static_assert(!std::assignable_from<NonAssignable, int>);

constexpr bool test()
{
  // Assigning from an lvalue.
  {
    std::move_sentinel<int> m(42);
    std::move_sentinel<long> m2;
    m2 = m;
    assert(m2.base() == 42L);
  }

  // Assigning from an rvalue.
  {
    std::move_sentinel<long> m2;
    m2 = std::move_sentinel<int>(43);
    assert(m2.base() == 43L);
  }

  // SFINAE checks.
  {
    static_assert( std::is_assignable_v<std::move_sentinel<int>, std::move_sentinel<long>>);
    static_assert(!std::is_assignable_v<std::move_sentinel<int*>, std::move_sentinel<const int*>>);
    static_assert( std::is_assignable_v<std::move_sentinel<const int*>, std::move_sentinel<int*>>);
    static_assert(!std::is_assignable_v<std::move_sentinel<NonAssignable>, std::move_sentinel<int>>);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
