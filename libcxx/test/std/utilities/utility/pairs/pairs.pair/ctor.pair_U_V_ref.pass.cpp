//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <utility>

// template <class T1, class T2> struct pair
// template <class U1, class U2>
//   constexpr explicit(see below) pair(pair<U1, U2>& p);

#include <cassert>
#include <utility>

#include "copy_move_types.h"
#include "test_macros.h"

// Constraints:
// is_constructible_v<T1, decltype(get<0>(FWD(p)))> is true and
// is_constructible_v<T2, decltype(get<1>(FWD(p)))> is true.
struct X {};
struct Y {};
struct NotConvertibleToXorY {};

// clang-format off
static_assert( std::is_constructible_v<std::pair<X, Y>, std::pair<X, Y>&>);
static_assert(!std::is_constructible_v<std::pair<X, Y>, std::pair<NotConvertibleToXorY, Y>&>);
static_assert(!std::is_constructible_v<std::pair<X, Y>, std::pair<X, NotConvertibleToXorY>&>);
static_assert(!std::is_constructible_v<std::pair<X, Y>, std::pair<NotConvertibleToXorY, NotConvertibleToXorY>&>);

// The expression inside explicit is equivalent to:
// !is_convertible_v<decltype(get<0>(FWD(p))), first_type> ||
// !is_convertible_v<decltype(get<1>(FWD(p))), second_type>.
static_assert( std::is_convertible_v<std::pair<X, Y>&, std::pair<ConvertibleFrom<X>, ConvertibleFrom<Y>>>);
static_assert(!std::is_convertible_v<std::pair<X, Y>&, std::pair<ConvertibleFrom<X>, ExplicitConstructibleFrom<Y>>>);
static_assert(!std::is_convertible_v<std::pair<X, Y>&, std::pair<ExplicitConstructibleFrom<X>, ConvertibleFrom<Y>>>);
static_assert(!std::is_convertible_v<std::pair<X, Y>&, std::pair<ExplicitConstructibleFrom<X>, ExplicitConstructibleFrom<Y>>>);
// clang-format on

constexpr bool test() {
  // use case in zip. Init pair<T&, U&> from pair<T, U>&
  {
    std::pair<int, int> p1{1, 2};
    std::pair<int&, int&> p2{p1};
    assert(&(p2.first) == &(p1.first));
    assert(&(p2.second) == &(p1.second));
  }

  // test implicit conversions.
  {
    std::pair<MutableCopy, int> p1{1, 2};
    std::pair<ConvertibleFrom<MutableCopy>, ConvertibleFrom<int>> p2 = p1;
    assert(p2.first.v.val == 1);
    assert(p2.second.v == 2);
  }

  // test explicit conversions.
  {
    std::pair<MutableCopy, int> p1{1, 2};
    std::pair<ExplicitConstructibleFrom<MutableCopy>, ExplicitConstructibleFrom<int>> p2{p1};
    assert(p2.first.v.val == 1);
    assert(p2.second.v == 2);
  }

  // test correct constructors of underlying types are called
  {
    std::pair<TracedCopyMove, TracedCopyMove> p1{};
    std::pair<ConvertibleFrom<TracedCopyMove>, ConvertibleFrom<TracedCopyMove>> p2{p1};
    assert(nonConstCopyCtrCalled(p2.first.v));
    assert(nonConstCopyCtrCalled(p2.second.v));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
