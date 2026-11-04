//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// constant_wrapper pseudo-mutators

// template<constexpr-param T>
//   constexpr auto operator++(this T) noexcept
//     -> constant_wrapper<++(T::value)> { return {}; }
// template<constexpr-param T>
//   constexpr auto operator++(this T, int) noexcept
//     -> constant_wrapper<(T::value++)> { return {}; }
// template<constexpr-param T>
//   constexpr auto operator--(this T) noexcept
//     -> constant_wrapper<--(T::value)> { return {}; }
// template<constexpr-param T>
//   constexpr auto operator--(this T, int) noexcept
//     -> constant_wrapper<(T::value--)> { return {}; }

// template<constexpr-param T, constexpr-param R>
//   constexpr auto operator+=(this T, R) noexcept
//     -> constant_wrapper<(T::value += R::value)> { return {}; }
// template<constexpr-param T, constexpr-param R>
//   constexpr auto operator-=(this T, R) noexcept
//     -> constant_wrapper<(T::value -= R::value)> { return {}; }
// template<constexpr-param T, constexpr-param R>
//   constexpr auto operator*=(this T, R) noexcept
//     -> constant_wrapper<(T::value *= R::value)> { return {}; }
// template<constexpr-param T, constexpr-param R>
//   constexpr auto operator/=(this T, R) noexcept
//     -> constant_wrapper<(T::value /= R::value)> { return {}; }
// template<constexpr-param T, constexpr-param R>
//   constexpr auto operator%=(this T, R) noexcept
//     -> constant_wrapper<(T::value %= R::value)> { return {}; }
// template<constexpr-param T, constexpr-param R>
//   constexpr auto operator&=(this T, R) noexcept
//     -> constant_wrapper<(T::value &= R::value)> { return {}; }
// template<constexpr-param T, constexpr-param R>
//   constexpr auto operator|=(this T, R) noexcept
//     -> constant_wrapper<(T::value |= R::value)> { return {}; }
// template<constexpr-param T, constexpr-param R>
//   constexpr auto operator^=(this T, R) noexcept
//     -> constant_wrapper<(T::value ^= R::value)> { return {}; }
// template<constexpr-param T, constexpr-param R>
//   constexpr auto operator<<=(this T, R) noexcept
//     -> constant_wrapper<(T::value <<= R::value)> { return {}; }
// template<constexpr-param T, constexpr-param R>
//   constexpr auto operator>>=(this T, R) noexcept
//     -> constant_wrapper<(T::value >>= R::value)> { return {}; }

#include <cassert>
#include <concepts>
#include <utility>

#include "helpers.h"

struct WithOps {
  int value;

  constexpr WithOps(int v) : value(v) {}

  constexpr auto operator++() const { return WithOps{value + 1}; }
  constexpr auto operator++(int) const { return WithOps{value + 1}; }
  constexpr auto operator--() const { return WithOps{value - 1}; }
  constexpr auto operator--(int) const { return WithOps{value - 1}; }

  constexpr auto operator+=(WithOps r) const { return WithOps{value + r.value}; }
  constexpr auto operator-=(WithOps r) const { return WithOps{value - r.value}; }
  constexpr auto operator*=(WithOps r) const { return WithOps{value * r.value}; }
  constexpr auto operator/=(WithOps r) const { return WithOps{value / r.value}; }
  constexpr auto operator%=(WithOps r) const { return WithOps{value % r.value}; }
  constexpr auto operator&=(WithOps r) const { return WithOps{value & r.value}; }
  constexpr auto operator|=(WithOps r) const { return WithOps{value | r.value}; }
  constexpr auto operator^=(WithOps r) const { return WithOps{value ^ r.value}; }
  constexpr auto operator<<=(WithOps r) const { return WithOps{value << r.value}; }
  constexpr auto operator>>=(WithOps r) const { return WithOps{value >> r.value}; }
};

struct OpsReturnNonStructural {
  int value;

  constexpr OpsReturnNonStructural(int v) : value(v) {}

  constexpr auto operator++() const { return NonStructural{value + 1}; }
  constexpr auto operator++(int) const { return NonStructural{value + 1}; }
  constexpr auto operator--() const { return NonStructural{value - 1}; }
  constexpr auto operator--(int) const { return NonStructural{value - 1}; }

  constexpr auto operator+=(OpsReturnNonStructural r) const { return NonStructural{value + r.value}; }
  constexpr auto operator-=(OpsReturnNonStructural r) const { return NonStructural{value - r.value}; }
  constexpr auto operator*=(OpsReturnNonStructural r) const { return NonStructural{value * r.value}; }
  constexpr auto operator/=(OpsReturnNonStructural r) const { return NonStructural{value / r.value}; }
  constexpr auto operator%=(OpsReturnNonStructural r) const { return NonStructural{value % r.value}; }
  constexpr auto operator&=(OpsReturnNonStructural r) const { return NonStructural{value & r.value}; }
  constexpr auto operator|=(OpsReturnNonStructural r) const { return NonStructural{value | r.value}; }
  constexpr auto operator^=(OpsReturnNonStructural r) const { return NonStructural{value ^ r.value}; }
  constexpr auto operator<<=(OpsReturnNonStructural r) const { return NonStructural{value << r.value}; }
  constexpr auto operator>>=(OpsReturnNonStructural r) const { return NonStructural{value >> r.value}; }
};

struct NoOps {};

template <class T>
concept HasPreIncrement = requires(T t) {
  { ++t };
};

template <class T>
concept HasPostIncrement = requires(T t) {
  { t++ };
};

template <class T>
concept HasPreDecrement = requires(T t) {
  { --t };
};

template <class T>
concept HasPostDecrement = requires(T t) {
  { t-- };
};

template <class L, class R>
concept HasPlusAssign = requires(L l, R r) {
  { l += r };
};

template <class L, class R>
concept HasMinusAssign = requires(L l, R r) {
  { l -= r };
};

template <class L, class R>
concept HasMultiplyAssign = requires(L l, R r) {
  { l *= r };
};

template <class L, class R>
concept HasDivideAssign = requires(L l, R r) {
  { l /= r };
};

template <class L, class R>
concept HasModuloAssign = requires(L l, R r) {
  { l %= r };
};

template <class L, class R>
concept HasBitAndAssign = requires(L l, R r) {
  { l &= r };
};

template <class L, class R>
concept HasBitOrAssign = requires(L l, R r) {
  { l |= r };
};

template <class L, class R>
concept HasBitXorAssign = requires(L l, R r) {
  { l ^= r };
};

template <class L, class R>
concept HasShiftLeftAssign = requires(L l, R r) {
  { l <<= r };
};

template <class L, class R>
concept HasShiftRightAssign = requires(L l, R r) {
  { l >>= r };
};

template <class T>
concept HasNoexceptPreIncrement = requires(T t) {
  { ++t } noexcept;
};

template <class T>
concept HasNoexceptPostIncrement = requires(T t) {
  { t++ } noexcept;
};

template <class T>
concept HasNoexceptPreDecrement = requires(T t) {
  { --t } noexcept;
};

template <class T>
concept HasNoexceptPostDecrement = requires(T t) {
  { t-- } noexcept;
};

template <class L, class R>
concept HasNoexceptPlusAssign = requires(L l, R r) {
  { l += r } noexcept;
};

template <class L, class R>
concept HasNoexceptMinusAssign = requires(L l, R r) {
  { l -= r } noexcept;
};

template <class L, class R>
concept HasNoexceptMultiplyAssign = requires(L l, R r) {
  { l *= r } noexcept;
};

template <class L, class R>
concept HasNoexceptDivideAssign = requires(L l, R r) {
  { l /= r } noexcept;
};

template <class L, class R>
concept HasNoexceptModuloAssign = requires(L l, R r) {
  { l %= r } noexcept;
};

template <class L, class R>
concept HasNoexceptBitAndAssign = requires(L l, R r) {
  { l &= r } noexcept;
};

template <class L, class R>
concept HasNoexceptBitOrAssign = requires(L l, R r) {
  { l |= r } noexcept;
};

template <class L, class R>
concept HasNoexceptBitXorAssign = requires(L l, R r) {
  { l ^= r } noexcept;
};

template <class L, class R>
concept HasNoexceptShiftLeftAssign = requires(L l, R r) {
  { l <<= r } noexcept;
};

template <class L, class R>
concept HasNoexceptShiftRightAssign = requires(L l, R r) {
  { l >>= r } noexcept;
};

// Pseudo-mutators does work with int as built-in types mutating operators are const
static_assert(!HasPreIncrement<std::constant_wrapper<6>>);
static_assert(!HasPostIncrement<std::constant_wrapper<6>>);
static_assert(!HasPreDecrement<std::constant_wrapper<6>>);
static_assert(!HasPostDecrement<std::constant_wrapper<6>>);

static_assert(!HasPlusAssign<std::constant_wrapper<6>, std::constant_wrapper<3>>);
static_assert(!HasMinusAssign<std::constant_wrapper<6>, std::constant_wrapper<3>>);
static_assert(!HasMultiplyAssign<std::constant_wrapper<6>, std::constant_wrapper<3>>);
static_assert(!HasDivideAssign<std::constant_wrapper<6>, std::constant_wrapper<3>>);
static_assert(!HasModuloAssign<std::constant_wrapper<6>, std::constant_wrapper<3>>);
static_assert(!HasBitAndAssign<std::constant_wrapper<6>, std::constant_wrapper<3>>);
static_assert(!HasBitOrAssign<std::constant_wrapper<6>, std::constant_wrapper<3>>);
static_assert(!HasBitXorAssign<std::constant_wrapper<6>, std::constant_wrapper<3>>);
static_assert(!HasShiftLeftAssign<std::constant_wrapper<6>, std::constant_wrapper<1>>);
static_assert(!HasShiftRightAssign<std::constant_wrapper<6>, std::constant_wrapper<1>>);

// NoOps - pseudo-mutators shouldn't work without supporting operators
static_assert(!HasPreIncrement<std::constant_wrapper<NoOps{}>>);
static_assert(!HasPostIncrement<std::constant_wrapper<NoOps{}>>);
static_assert(!HasPreDecrement<std::constant_wrapper<NoOps{}>>);
static_assert(!HasPostDecrement<std::constant_wrapper<NoOps{}>>);

static_assert(!HasPlusAssign<std::constant_wrapper<NoOps{}>, std::constant_wrapper<NoOps{}>>);
static_assert(!HasMinusAssign<std::constant_wrapper<NoOps{}>, std::constant_wrapper<NoOps{}>>);
static_assert(!HasMultiplyAssign<std::constant_wrapper<NoOps{}>, std::constant_wrapper<NoOps{}>>);
static_assert(!HasDivideAssign<std::constant_wrapper<NoOps{}>, std::constant_wrapper<NoOps{}>>);
static_assert(!HasModuloAssign<std::constant_wrapper<NoOps{}>, std::constant_wrapper<NoOps{}>>);
static_assert(!HasBitAndAssign<std::constant_wrapper<NoOps{}>, std::constant_wrapper<NoOps{}>>);
static_assert(!HasBitOrAssign<std::constant_wrapper<NoOps{}>, std::constant_wrapper<NoOps{}>>);
static_assert(!HasBitXorAssign<std::constant_wrapper<NoOps{}>, std::constant_wrapper<NoOps{}>>);
static_assert(!HasShiftLeftAssign<std::constant_wrapper<NoOps{}>, std::constant_wrapper<NoOps{}>>);
static_assert(!HasShiftRightAssign<std::constant_wrapper<NoOps{}>, std::constant_wrapper<NoOps{}>>);

// Pseudo-mutators work with WithOps types
static_assert(HasNoexceptPreIncrement<std::constant_wrapper<WithOps{6}>>);
static_assert(HasNoexceptPostIncrement<std::constant_wrapper<WithOps{6}>>);
static_assert(HasNoexceptPreDecrement<std::constant_wrapper<WithOps{6}>>);
static_assert(HasNoexceptPostDecrement<std::constant_wrapper<WithOps{6}>>);

static_assert(HasNoexceptPlusAssign<std::constant_wrapper<WithOps{6}>, std::constant_wrapper<WithOps{3}>>);
static_assert(HasNoexceptMinusAssign<std::constant_wrapper<WithOps{6}>, std::constant_wrapper<WithOps{3}>>);
static_assert(HasNoexceptMultiplyAssign<std::constant_wrapper<WithOps{6}>, std::constant_wrapper<WithOps{3}>>);
static_assert(HasNoexceptDivideAssign<std::constant_wrapper<WithOps{6}>, std::constant_wrapper<WithOps{3}>>);
static_assert(HasNoexceptModuloAssign<std::constant_wrapper<WithOps{6}>, std::constant_wrapper<WithOps{3}>>);
static_assert(HasNoexceptBitAndAssign<std::constant_wrapper<WithOps{6}>, std::constant_wrapper<WithOps{3}>>);
static_assert(HasNoexceptBitOrAssign<std::constant_wrapper<WithOps{6}>, std::constant_wrapper<WithOps{3}>>);
static_assert(HasNoexceptBitXorAssign<std::constant_wrapper<WithOps{6}>, std::constant_wrapper<WithOps{3}>>);
static_assert(HasNoexceptShiftLeftAssign<std::constant_wrapper<WithOps{6}>, std::constant_wrapper<WithOps{1}>>);
static_assert(HasNoexceptShiftRightAssign<std::constant_wrapper<WithOps{6}>, std::constant_wrapper<WithOps{1}>>);

// clang-format off
// Non-structural return types cannot use implicit conversions too because they are member functions and cannot be found through ADL
static_assert(!HasPreIncrement<std::constant_wrapper<OpsReturnNonStructural{6}>>);
static_assert(!HasPostIncrement<std::constant_wrapper<OpsReturnNonStructural{6}>>);
static_assert(!HasPreDecrement<std::constant_wrapper<OpsReturnNonStructural{6}>>);
static_assert(!HasPostDecrement<std::constant_wrapper<OpsReturnNonStructural{6}>>);

static_assert(!HasPlusAssign<std::constant_wrapper<OpsReturnNonStructural{6}>, std::constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasMinusAssign<std::constant_wrapper<OpsReturnNonStructural{6}>, std::constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasMultiplyAssign<std::constant_wrapper<OpsReturnNonStructural{6}>, std::constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasDivideAssign<std::constant_wrapper<OpsReturnNonStructural{6}>, std::constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasModuloAssign<std::constant_wrapper<OpsReturnNonStructural{6}>, std::constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasBitAndAssign<std::constant_wrapper<OpsReturnNonStructural{6}>, std::constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasBitOrAssign<std::constant_wrapper<OpsReturnNonStructural{6}>, std::constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasBitXorAssign<std::constant_wrapper<OpsReturnNonStructural{6}>, std::constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasShiftLeftAssign<std::constant_wrapper<OpsReturnNonStructural{6}>, std::constant_wrapper<OpsReturnNonStructural{1}>>);
static_assert(!HasShiftRightAssign<std::constant_wrapper<OpsReturnNonStructural{6}>, std::constant_wrapper<OpsReturnNonStructural{1}>>);
// clang-format on

// LWG 4383. constant_wrapper's pseudo-mutators are underconstrained
// https://cplusplus.github.io/LWG/issue4383
constexpr void lwg4383_f(auto t) {
  if constexpr (requires { +t; }) // ok
    +t;
  if constexpr (requires { -t; }) // ok
    -t;
  if constexpr (requires { ++t; }) // no hard error
    ++t;
  if constexpr (requires { --t; }) // no hard error
    --t;
}

struct S {
  /* constexpr */ int operator+() const { return 0; }
  /* constexpr */ int operator++() { return 0; }
  constexpr void operator-() const {}
  constexpr void operator--() {}
};

constexpr void lwg4383() { lwg4383_f(std::cw<S{}>); }

constexpr bool test() {
  {
    // WithOps increment/decrement
    std::constant_wrapper<WithOps{5}> cwWithOps5;
    std::same_as<std::constant_wrapper<WithOps{6}>> decltype(auto) result1 = ++cwWithOps5;
    static_assert(result1.value.value == 6);

    std::same_as<std::constant_wrapper<WithOps{6}>> decltype(auto) result2 = cwWithOps5++;
    static_assert(result2.value.value == 6);

    std::same_as<std::constant_wrapper<WithOps{4}>> decltype(auto) result3 = --cwWithOps5;
    static_assert(result3.value.value == 4);

    std::same_as<std::constant_wrapper<WithOps{4}>> decltype(auto) result4 = cwWithOps5--;
    static_assert(result4.value.value == 4);
  }

  {
    // WithOps compound assignments
    std::constant_wrapper<WithOps{10}> cwWithOps10;
    std::constant_wrapper<WithOps{3}> cwWithOps3;

    std::same_as<std::constant_wrapper<WithOps{13}>> decltype(auto) result1 = cwWithOps10 += cwWithOps3;
    static_assert(result1.value.value == 13);

    std::same_as<std::constant_wrapper<WithOps{7}>> decltype(auto) result2 = cwWithOps10 -= cwWithOps3;
    static_assert(result2.value.value == 7);

    std::same_as<std::constant_wrapper<WithOps{30}>> decltype(auto) result3 = cwWithOps10 *= cwWithOps3;
    static_assert(result3.value.value == 30);

    std::same_as<std::constant_wrapper<WithOps{3}>> decltype(auto) result4 = cwWithOps10 /= cwWithOps3;
    static_assert(result4.value.value == 3);

    std::same_as<std::constant_wrapper<WithOps{1}>> decltype(auto) result5 = cwWithOps10 %= cwWithOps3;
    static_assert(result5.value.value == 1);

    std::same_as<std::constant_wrapper<WithOps{2}>> decltype(auto) result6 = cwWithOps10 &= cwWithOps3;
    static_assert(result6.value.value == 2);

    std::same_as<std::constant_wrapper<WithOps{11}>> decltype(auto) result7 = cwWithOps10 |= cwWithOps3;
    static_assert(result7.value.value == 11);

    std::same_as<std::constant_wrapper<WithOps{9}>> decltype(auto) result8 = cwWithOps10 ^= cwWithOps3;
    static_assert(result8.value.value == 9);

    std::same_as<std::constant_wrapper<WithOps{80}>> decltype(auto) result9 = cwWithOps10 <<= cwWithOps3;
    static_assert(result9.value.value == 80);

    std::same_as<std::constant_wrapper<WithOps{1}>> decltype(auto) result10 = cwWithOps10 >>= cwWithOps3;
    static_assert(result10.value.value == 1);
  }

  {
    // integral_constant compound assignments
    std::constant_wrapper<WithOps{10}> cwWithOps10;
    std::integral_constant<WithOps, WithOps{3}> icWithOps3;

    std::same_as<std::constant_wrapper<WithOps{13}>> decltype(auto) result1 = cwWithOps10 += icWithOps3;
    static_assert(result1.value.value == 13);

    std::same_as<std::constant_wrapper<WithOps{7}>> decltype(auto) result2 = cwWithOps10 -= icWithOps3;
    static_assert(result2.value.value == 7);

    std::same_as<std::constant_wrapper<WithOps{30}>> decltype(auto) result3 = cwWithOps10 *= icWithOps3;
    static_assert(result3.value.value == 30);

    std::same_as<std::constant_wrapper<WithOps{3}>> decltype(auto) result4 = cwWithOps10 /= icWithOps3;
    static_assert(result4.value.value == 3);

    std::same_as<std::constant_wrapper<WithOps{1}>> decltype(auto) result5 = cwWithOps10 %= icWithOps3;
    static_assert(result5.value.value == 1);

    std::same_as<std::constant_wrapper<WithOps{2}>> decltype(auto) result6 = cwWithOps10 &= icWithOps3;
    static_assert(result6.value.value == 2);

    std::same_as<std::constant_wrapper<WithOps{11}>> decltype(auto) result7 = cwWithOps10 |= icWithOps3;
    static_assert(result7.value.value == 11);

    std::same_as<std::constant_wrapper<WithOps{9}>> decltype(auto) result8 = cwWithOps10 ^= icWithOps3;
    static_assert(result8.value.value == 9);

    std::same_as<std::constant_wrapper<WithOps{80}>> decltype(auto) result9 = cwWithOps10 <<= icWithOps3;
    static_assert(result9.value.value == 80);

    std::same_as<std::constant_wrapper<WithOps{1}>> decltype(auto) result10 = cwWithOps10 >>= icWithOps3;
    static_assert(result10.value.value == 1);
  }

  lwg4383();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
