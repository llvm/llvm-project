//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// constant_wrapper

// template<class... Args>
// static constexpr decltype(auto) operator[](Args&&... args) noexcept(see below);

#include <cassert>
#include <concepts>
#include <utility>

#include "helpers.h"
#include "MoveOnly.h"

struct MoveOnlyIndex {
  constexpr MoveOnly operator[](const MoveOnly& m1, MoveOnly m2, MoveOnly&& m3) const {
    return MoveOnly(m1.get() + m2.get() + m3.get());
  }
};

struct Nary {
  constexpr int operator[](auto... args) const { return sizeof...(args); }
};

struct OverloadSet {
  constexpr int operator[](int) const { return 1; }

  constexpr int operator[](std::constant_wrapper<42>) const { return 2; }
};

struct ReturnNonStructural {
  constexpr NonStructural operator[](int i) const { return NonStructural{i}; }
};

struct CWOnly {
  constexpr int operator[](std::constant_wrapper<42>) const { return 42; }
};

struct ThrowingSubscript {
  constexpr int operator[](int) const { return 42; }
};

struct NothrowSubscript {
  constexpr int operator[](int) const noexcept { return 42; }
};

constexpr int arr[] = {1, 2, 3, 4};

// Let subscr-expr be constant_wrapper<value[remove_cvref_t<Args>::value...]>{} if all types in remove_cvref_t<Args>... satisfy constexpr-param and constant_wrapper<value[remove_cvref_t<Args>::value...]>
// is a valid type, otherwise let subscr-expr be value[std::forward<Args>(args)...].
// - Constraints: subscr-expr is a valid expression.
// - Remarks: The exception specification is equivalent to noexcept(subscr-expr).

template <class T, class... Args>
concept HasSubscript = requires(T t, Args&&... args) {
  { t[std::forward<Args>(args)...] };
};

template <class T, class... Args>
concept HasNothrowSubscript = requires(T t, Args&&... args) {
  { t[std::forward<Args>(args)...] } noexcept;
};

static_assert(!HasSubscript<std::constant_wrapper<4>, std::constant_wrapper<1>>);

static_assert(HasSubscript<std::constant_wrapper<arr>, int>);
static_assert(HasSubscript<std::constant_wrapper<arr>, std::constant_wrapper<1>>);

static_assert(HasNothrowSubscript<std::constant_wrapper<arr>, int>);
static_assert(HasNothrowSubscript<std::constant_wrapper<arr>, std::constant_wrapper<1>>);

static_assert(HasSubscript<std::constant_wrapper<NothrowSubscript{}>, int>);
static_assert(HasNothrowSubscript<std::constant_wrapper<NothrowSubscript{}>, int>);

static_assert(HasSubscript<std::constant_wrapper<ThrowingSubscript{}>, int>);
static_assert(!HasNothrowSubscript<std::constant_wrapper<ThrowingSubscript{}>, int>);
static_assert(HasNothrowSubscript<std::constant_wrapper<ThrowingSubscript{}>, std::constant_wrapper<1>>,
              "the subscript expression is still nothrow because the constexpr path is taken");

constexpr bool test() {
  {
    // with runtime param
    using T                                        = std::constant_wrapper<arr>;
    std::same_as<const int&> decltype(auto) result = T::operator[](1);
    assert(result == 2);
  }
  {
    // with constexpr param
    using T                                                      = std::constant_wrapper<arr>;
    std::same_as<std::constant_wrapper<2>> decltype(auto) result = T::operator[](std::cw<1>);
    static_assert(result == 2);
  }

  {
    // null-ary
    using T                                                      = std::constant_wrapper<Nary{}>;
    std::same_as<std::constant_wrapper<0>> decltype(auto) result = T::operator[]();
    static_assert(result == 0);
  }

  {
    // n-ary
    using T                                                      = std::constant_wrapper<Nary{}>;
    std::same_as<std::constant_wrapper<3>> decltype(auto) result = T::operator[](std::cw<1>, std::cw<2>, std::cw<3>);
    static_assert(result == 3);
  }

  {
    // mixing constexpr and runtime
    using T                                 = std::constant_wrapper<Nary{}>;
    std::same_as<int> decltype(auto) result = T::operator[](std::cw<1>, 2, std::cw<3>);
    assert(result == 3);
  }

  {
    // move only
    using T = std::constant_wrapper<MoveOnlyIndex{}>;
    MoveOnly m1(1), m2(2), m3(3);
    std::same_as<MoveOnly> decltype(auto) result = T::operator[](m1, std::move(m2), std::move(m3));
    assert(result.get() == 6);
  }
  {
    // overload set
    // will always unwrap the constexpr params and call the non-constexpr overload
    using T                                  = std::constant_wrapper<OverloadSet{}>;
    std::same_as<int> decltype(auto) result1 = T::operator[](42);
    assert(result1 == 1);
    std::same_as<std::constant_wrapper<1>> decltype(auto) result2 = T::operator[](std::cw<42>);
    static_assert(result2 == 1);
  }

  {
    // return non-structural type
    using T                                           = std::constant_wrapper<ReturnNonStructural{}>;
    std::same_as<NonStructural> decltype(auto) result = T::operator[](5);
    assert(result.get() == 5);
  }

  {
    // return non-structural type with constexpr param
    using T                                           = std::constant_wrapper<ReturnNonStructural{}>;
    std::same_as<NonStructural> decltype(auto) result = T::operator[](std::cw<5>);
    assert(result.get() == 5);
  }

  {
    // cw only
    // the upwrapping case doesn't work so it falls back to the normal invoke path
    using T                                 = std::constant_wrapper<CWOnly{}>;
    std::same_as<int> decltype(auto) result = T::operator[](std::cw<42>);
    assert(result == 42);
  }

  {
    // just use the index operator
    assert(std::cw<"abcd">[2] == 'c');
    assert(std::cw<"abcd">[std::cw<3>] == 'd');
  }

  {
    // integral_constant
    using T                                                      = std::constant_wrapper<arr>;
    std::same_as<std::constant_wrapper<2>> decltype(auto) result = T::operator[](std::integral_constant<int, 1>{});
    static_assert(result == 2);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
