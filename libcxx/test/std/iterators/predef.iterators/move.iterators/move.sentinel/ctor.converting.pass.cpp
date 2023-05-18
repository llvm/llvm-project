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
//    requires convertible_to<const S2&, S>
//      constexpr move_sentinel(const move_sentinel<S2>& s);

#include <cassert>
#include <concepts>
#include <iterator>
#include <type_traits>

struct NonConvertible {
    explicit NonConvertible();
    NonConvertible(int i);
    explicit NonConvertible(long i) = delete;
};
static_assert(std::semiregular<NonConvertible>);
static_assert(std::is_convertible_v<long, NonConvertible>);
static_assert(!std::convertible_to<long, NonConvertible>);

constexpr bool test()
{
  // Constructing from an lvalue.
  {
    std::move_sentinel<int> m(42);
    std::move_sentinel<long> m2 = m;
    assert(m2.base() == 42L);
  }

  // Constructing from an rvalue.
  {
    std::move_sentinel<long> m2 = std::move_sentinel<int>(43);
    assert(m2.base() == 43L);
  }

  // SFINAE checks.
  {
    static_assert( std::is_convertible_v<std::move_sentinel<int>, std::move_sentinel<long>>);
    static_assert( std::is_convertible_v<std::move_sentinel<int*>, std::move_sentinel<const int*>>);
    static_assert(!std::is_convertible_v<std::move_sentinel<const int*>, std::move_sentinel<int*>>);
    static_assert( std::is_convertible_v<std::move_sentinel<int>, std::move_sentinel<NonConvertible>>);
    static_assert(!std::is_convertible_v<std::move_sentinel<long>, std::move_sentinel<NonConvertible>>);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
