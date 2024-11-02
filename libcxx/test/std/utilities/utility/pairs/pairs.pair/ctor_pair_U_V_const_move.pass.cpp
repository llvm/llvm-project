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
//   constexpr explicit(see below) pair(const pair<U1, U2>&& p);

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

static_assert(std::is_constructible_v<std::pair<X, Y>, const std::pair<X, Y>&&>);
static_assert(!std::is_constructible_v<std::pair<X, Y>, const std::pair<NotConvertibleToXorY, Y>&&>);
static_assert(!std::is_constructible_v<std::pair<X, Y>, const std::pair<X, NotConvertibleToXorY>&&>);
static_assert(!std::is_constructible_v<std::pair<X, Y>, const std::pair<NotConvertibleToXorY, NotConvertibleToXorY>&&>);

// The expression inside explicit is equivalent to:
// !is_convertible_v<decltype(get<0>(FWD(p))), first_type> ||
// !is_convertible_v<decltype(get<1>(FWD(p))), second_type>.
// clang-format off
static_assert( std::is_convertible_v<const std::pair<X, Y>&&,
                                     std::pair<ConvertibleFrom<X>, ConvertibleFrom<Y>>>);
static_assert(!std::is_convertible_v<const std::pair<X, Y>&&,
                                     std::pair<ConvertibleFrom<X>, ExplicitConstructibleFrom<Y>>>);
static_assert(!std::is_convertible_v<const std::pair<X, Y>&&,
                                     std::pair<ExplicitConstructibleFrom<X>, ConvertibleFrom<Y>>>);
static_assert(!std::is_convertible_v<const std::pair<X, Y>&&,
                                     std::pair<ExplicitConstructibleFrom<X>, ExplicitConstructibleFrom<Y>>>);
// clang-format on

constexpr bool test() {
  // simple case: init pair<const T&&, const U&&> from const pair<T, U>&&
  {
    const std::pair<int, int> p1{1, 2};
    std::pair<const int&&, const int&&> p2{std::move(p1)};
    assert(&(p2.first) == &(p1.first));
    assert(&(p2.second) == &(p1.second));
  }

  // test implicit conversions.
  {
    const std::pair<ConstMove, int> p1{1, 2};
    std::pair<ConvertibleFrom<ConstMove>, ConvertibleFrom<int>> p2 = std::move(p1);
    assert(p2.first.v.val == 1);
    assert(p2.second.v == 2);
  }

  // test explicit conversions.
  {
    const std::pair<ConstMove, int> p1{1, 2};
    std::pair<ExplicitConstructibleFrom<ConstMove>, ExplicitConstructibleFrom<int>> p2{std::move(p1)};
    assert(p2.first.v.val == 1);
    assert(p2.second.v == 2);
  }

  // test correct constructors of underlying types are called
  {
    const std::pair<TracedCopyMove, TracedCopyMove> p1{};
    std::pair<ConvertibleFrom<TracedCopyMove>, ConvertibleFrom<TracedCopyMove>> p2{std::move(p1)};
    assert(constMoveCtrCalled(p2.first.v));
    assert(constMoveCtrCalled(p2.second.v));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
