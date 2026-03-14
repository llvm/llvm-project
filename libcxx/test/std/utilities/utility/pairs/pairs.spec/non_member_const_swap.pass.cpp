//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2>
// void swap(const pair<T1, T2>& x, const pair<T1, T2>& y) const noexcept(noexcept(x.swap(y)));;

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <cassert>
#include <type_traits>
#include <utility>

#include "test_macros.h"

// Constraints:
// For the second overload, is_swappable_v<const T1> is true and is_swappable_v<const T2> is true.
struct NonConstSwappable {
  friend constexpr void swap(NonConstSwappable&, NonConstSwappable&);
};

struct ConstSwappable {
  mutable int i;
  friend constexpr void swap(const ConstSwappable& lhs, const ConstSwappable& rhs) { std::swap(lhs.i, rhs.i); }
};

static_assert(std::is_swappable_v<const std::pair<ConstSwappable, ConstSwappable>>);
static_assert(!std::is_swappable_v<const std::pair<NonConstSwappable, ConstSwappable>>);
static_assert(!std::is_swappable_v<const std::pair<ConstSwappable, NonConstSwappable>>);
static_assert(!std::is_swappable_v<const std::pair<NonConstSwappable, NonConstSwappable>>);

// noexcept(noexcept(x.swap(y)));
template <class T>
concept NonMemberSwapNoexcept =
    requires(T t1, T t2) {
      { swap(t1, t2) } noexcept;
    };

template <bool canThrow>
struct SwapMayThrow {};

template <bool canThrow>
void swap(const SwapMayThrow<canThrow>&, const SwapMayThrow<canThrow>&) noexcept(!canThrow);

static_assert(NonMemberSwapNoexcept<const std::pair<SwapMayThrow<false>, SwapMayThrow<false>>>);
static_assert(!NonMemberSwapNoexcept<const std::pair<SwapMayThrow<true>, SwapMayThrow<false>>>);
static_assert(!NonMemberSwapNoexcept<const std::pair<SwapMayThrow<false>, SwapMayThrow<true>>>);
static_assert(!NonMemberSwapNoexcept<const std::pair<SwapMayThrow<true>, SwapMayThrow<true>>>);

constexpr bool test() {
  // user defined const swap
  {
    using P = std::pair<const ConstSwappable, const ConstSwappable>;
    const P p1(ConstSwappable{0}, ConstSwappable{1});
    const P p2(ConstSwappable{2}, ConstSwappable{3});
    using std::swap;
    swap(p1, p2);
    assert(p1.first.i == 2);
    assert(p1.second.i == 3);
    assert(p2.first.i == 0);
    assert(p2.second.i == 1);
  }

  // pair of references
  {
    int i1 = 0, i2 = 1, i3 = 2, i4 = 3;
    const std::pair<int&, int&> p1{i1, i2};
    const std::pair<int&, int&> p2{i3, i4};
    using std::swap;
    swap(p1, p2);
    assert(p1.first == 2);
    assert(p1.second == 3);
    assert(p2.first == 0);
    assert(p2.second == 1);
  }
  return true;
}

int main(int, char**) {
  test();

// gcc cannot have mutable member in constant expression
#if !defined(TEST_COMPILER_GCC)
  static_assert(test());
#endif

  return 0;
}