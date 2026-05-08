//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// constant_wrapper

//  template<constexpr-param T>
//    friend constexpr auto operator+(T) noexcept -> constant_wrapper<(+T::value)>
//      { return {}; }
//  template<constexpr-param T>
//    friend constexpr auto operator-(T) noexcept -> constant_wrapper<(-T::value)>
//      { return {}; }
//  template<constexpr-param T>
//    friend constexpr auto operator~(T) noexcept -> constant_wrapper<(~T::value)>
//      { return {}; }
//  template<constexpr-param T>
//    friend constexpr auto operator!(T) noexcept -> constant_wrapper<(!T::value)>
//      { return {}; }
//  template<constexpr-param T>
//    friend constexpr auto operator&(T) noexcept -> constant_wrapper<(&T::value)>
//      { return {}; }
//  template<constexpr-param T>
//    friend constexpr auto operator*(T) noexcept -> constant_wrapper<(*T::value)>
//      { return {}; }

#include <cassert>
#include <concepts>
#include <utility>

#include "helpers.h"

struct WithOps {
  int value;

  constexpr WithOps(int v) : value(v) {}

  friend constexpr auto operator+(WithOps w) { return WithOps{+w.value}; }
  friend constexpr auto operator-(WithOps w) { return WithOps{-w.value}; }
  friend constexpr auto operator~(WithOps w) { return WithOps{~w.value}; }
  friend constexpr auto operator!(WithOps w) { return WithOps{!w.value}; }
  friend constexpr auto operator&(WithOps w) { return WithOps{w.value + 42}; }
  friend constexpr auto operator*(WithOps w) { return WithOps{w.value - 42}; }
};

struct OpsReturnNonStructural {
  int value;

  constexpr OpsReturnNonStructural(int v) : value(v) {}

  friend constexpr auto operator+(OpsReturnNonStructural o) { return NonStructural{+o.value}; }
  friend constexpr auto operator-(OpsReturnNonStructural o) { return NonStructural{-o.value}; }
  friend constexpr auto operator~(OpsReturnNonStructural o) { return NonStructural{~o.value}; }
  friend constexpr auto operator!(OpsReturnNonStructural o) { return NonStructural{!o.value}; }
  friend constexpr auto operator&(OpsReturnNonStructural o) { return NonStructural{o.value + 42}; }
  friend constexpr auto operator*(OpsReturnNonStructural o) { return NonStructural{o.value - 42}; }
};

struct NoOps {};

template <class T>
concept HasPlus = requires(T t) {
  { +t };
};

template <class T>
concept HasMinus = requires(T t) {
  { -t };
};

template <class T>
concept HasBitNot = requires(T t) {
  { ~t };
};

template <class T>
concept HasNot = requires(T t) {
  { !t };
};

template <class T>
concept HasBitAnd = requires(T t) {
  { &t };
};

template <class T>
concept HasDeref = requires(T t) {
  { *t };
};

template <class T>
concept HasNoexceptPlus = requires(T t) {
  { +t } noexcept;
};

template <class T>
concept HasNoexceptMinus = requires(T t) {
  { -t } noexcept;
};

template <class T>
concept HasNoexceptBitNot = requires(T t) {
  { ~t } noexcept;
};

template <class T>
concept HasNoexceptNot = requires(T t) {
  { !t } noexcept;
};

template <class T>
concept HasNoexceptBitAnd = requires(T t) {
  { &t } noexcept;
};

template <class T>
concept HasNoexceptDeref = requires(T t) {
  { *t } noexcept;
};

static_assert(HasPlus<std::constant_wrapper<WithOps{42}>>);
static_assert(HasMinus<std::constant_wrapper<WithOps{42}>>);
static_assert(HasBitNot<std::constant_wrapper<WithOps{42}>>);
static_assert(HasNot<std::constant_wrapper<WithOps{42}>>);
static_assert(HasBitAnd<std::constant_wrapper<WithOps{42}>>);
static_assert(HasDeref<std::constant_wrapper<WithOps{42}>>);

static_assert(HasNoexceptPlus<std::constant_wrapper<WithOps{42}>>);
static_assert(HasNoexceptMinus<std::constant_wrapper<WithOps{42}>>);
static_assert(HasNoexceptBitNot<std::constant_wrapper<WithOps{42}>>);
static_assert(HasNoexceptNot<std::constant_wrapper<WithOps{42}>>);
static_assert(HasNoexceptBitAnd<std::constant_wrapper<WithOps{42}>>);
static_assert(HasNoexceptDeref<std::constant_wrapper<WithOps{42}>>);

static_assert(HasNoexceptPlus<std::constant_wrapper<42>>);
static_assert(HasNoexceptMinus<std::constant_wrapper<42>>);
static_assert(HasNoexceptBitNot<std::constant_wrapper<42>>);
static_assert(HasNoexceptNot<std::constant_wrapper<42>>);
static_assert(HasNoexceptBitAnd<std::constant_wrapper<42>>);
static_assert(!HasDeref<std::constant_wrapper<42>>);

static_assert(!HasPlus<std::constant_wrapper<NoOps{}>>);
static_assert(!HasMinus<std::constant_wrapper<NoOps{}>>);
static_assert(!HasBitNot<std::constant_wrapper<NoOps{}>>);
static_assert(!HasNot<std::constant_wrapper<NoOps{}>>);
static_assert(HasBitAnd<std::constant_wrapper<NoOps{}>>);
static_assert(!HasDeref<std::constant_wrapper<NoOps{}>>);

// The operators from constant_wrapper do not exist, but they can be implicited converted
// to the underlying type and use its operators instead.
static_assert(HasPlus<std::constant_wrapper<OpsReturnNonStructural{42}>>);
static_assert(HasMinus<std::constant_wrapper<OpsReturnNonStructural{42}>>);
static_assert(HasBitNot<std::constant_wrapper<OpsReturnNonStructural{42}>>);
static_assert(HasNot<std::constant_wrapper<OpsReturnNonStructural{42}>>);
static_assert(HasBitAnd<std::constant_wrapper<OpsReturnNonStructural{42}>>);
static_assert(HasDeref<std::constant_wrapper<OpsReturnNonStructural{42}>>);

static_assert(!HasNoexceptPlus<std::constant_wrapper<OpsReturnNonStructural{42}>>);
static_assert(!HasNoexceptMinus<std::constant_wrapper<OpsReturnNonStructural{42}>>);
static_assert(!HasNoexceptBitNot<std::constant_wrapper<OpsReturnNonStructural{42}>>);
static_assert(!HasNoexceptNot<std::constant_wrapper<OpsReturnNonStructural{42}>>);
static_assert(!HasNoexceptBitAnd<std::constant_wrapper<OpsReturnNonStructural{42}>>);
static_assert(!HasNoexceptDeref<std::constant_wrapper<OpsReturnNonStructural{42}>>);

constexpr bool test() {
  {
    // int
    std::constant_wrapper<42> cw42;

    std::same_as<std::constant_wrapper<42>> decltype(auto) result = +cw42;
    static_assert(result == 42);

    std::same_as<std::constant_wrapper<-42>> decltype(auto) result2 = -cw42;
    static_assert(result2 == -42);

    std::same_as<std::constant_wrapper<~42>> decltype(auto) result3 = ~cw42;
    static_assert(result3 == ~42);

    std::same_as<std::constant_wrapper<!42>> decltype(auto) result4 = !cw42;
    static_assert(result4 == !42);

    std::same_as<std::constant_wrapper<&cw42.value>> decltype(auto) result5 = &cw42;
    static_assert(result5 == &cw42.value);
  }

  {
    // WithOps
    std::constant_wrapper<WithOps{42}> cwWithOps;

    std::same_as<std::constant_wrapper<WithOps{42}>> decltype(auto) result = +cwWithOps;
    static_assert(result.value.value == 42);

    std::same_as<std::constant_wrapper<WithOps{-42}>> decltype(auto) result2 = -cwWithOps;
    static_assert(result2.value.value == -42);

    std::same_as<std::constant_wrapper<WithOps{~42}>> decltype(auto) result3 = ~cwWithOps;
    static_assert(result3.value.value == ~42);

    std::same_as<std::constant_wrapper<WithOps{!42}>> decltype(auto) result4 = !cwWithOps;
    static_assert(result4.value.value == !42);

    std::same_as<std::constant_wrapper<WithOps{84}>> decltype(auto) result5 = &cwWithOps;
    static_assert(result5.value.value == 84);

    std::same_as<std::constant_wrapper<WithOps{0}>> decltype(auto) result6 = *cwWithOps;
    static_assert(result6.value.value == 0);
  }

  {
    // Return non-structural type
    // Will use underlying type's runtime operators
    std::constant_wrapper<OpsReturnNonStructural{42}> cwOpsReturnNonStructural;

    std::same_as<NonStructural> decltype(auto) result = +cwOpsReturnNonStructural;
    assert(result.get() == 42);

    std::same_as<NonStructural> decltype(auto) result2 = -cwOpsReturnNonStructural;
    assert(result2.get() == -42);

    std::same_as<NonStructural> decltype(auto) result3 = ~cwOpsReturnNonStructural;
    assert(result3.get() == ~42);

    std::same_as<NonStructural> decltype(auto) result4 = !cwOpsReturnNonStructural;
    assert(result4.get() == !42);

    std::same_as<NonStructural> decltype(auto) result5 = &cwOpsReturnNonStructural;
    assert(result5.get() == 84);

    std::same_as<NonStructural> decltype(auto) result6 = *cwOpsReturnNonStructural;
    assert(result6.get() == 0);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
