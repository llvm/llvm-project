//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// ADDITIONAL_COMPILE_FLAGS: -Wno-constant-logical-operand

// constant_wrapper

//  template<constexpr-param L, constexpr-param R>
//    friend constexpr auto operator+(L, R) noexcept -> constant_wrapper<(L::value + R::value)>
//      { return {}; }
//  template<constexpr-param L, constexpr-param R>
//    friend constexpr auto operator-(L, R) noexcept -> constant_wrapper<(L::value - R::value)>
//      { return {}; }
//  template<constexpr-param L, constexpr-param R>
//    friend constexpr auto operator*(L, R) noexcept -> constant_wrapper<(L::value * R::value)>
//      { return {}; }
//  template<constexpr-param L, constexpr-param R>
//    friend constexpr auto operator/(L, R) noexcept -> constant_wrapper<(L::value / R::value)>
//      { return {}; }
//  template<constexpr-param L, constexpr-param R>
//    friend constexpr auto operator%(L, R) noexcept -> constant_wrapper<(L::value % R::value)>
//      { return {}; }

//  template<constexpr-param L, constexpr-param R>
//    friend constexpr auto operator<<(L, R) noexcept -> constant_wrapper<(L::value << R::value)>
//      { return {}; }
//  template<constexpr-param L, constexpr-param R>
//    friend constexpr auto operator>>(L, R) noexcept -> constant_wrapper<(L::value >> R::value)>
//      { return {}; }
//  template<constexpr-param L, constexpr-param R>
//    friend constexpr auto operator&(L, R) noexcept -> constant_wrapper<(L::value & R::value)>
//      { return {}; }
//  template<constexpr-param L, constexpr-param R>
//    friend constexpr auto operator|(L, R) noexcept -> constant_wrapper<(L::value | R::value)>
//      { return {}; }
//  template<constexpr-param L, constexpr-param R>
//    friend constexpr auto operator^(L, R) noexcept -> constant_wrapper<(L::value ^ R::value)>
//      { return {}; }

//  template<constexpr-param L, constexpr-param R>
//    requires (!is_constructible_v<bool, decltype(L::value)> ||
//              !is_constructible_v<bool, decltype(R::value)>)
//      friend constexpr auto operator&&(L, R) noexcept
//        -> constant_wrapper<(L::value && R::value)>
//          { return {}; }
//  template<constexpr-param L, constexpr-param R>
//    requires (!is_constructible_v<bool, decltype(L::value)> ||
//              !is_constructible_v<bool, decltype(R::value)>)
//      friend constexpr auto operator||(L, R) noexcept
//        -> constant_wrapper<(L::value || R::value)>
//          { return {}; }

#include <cassert>
#include <concepts>
#include <type_traits>
#include <utility>

#include "helpers.h"

struct WithOps {
  int value;

  constexpr WithOps(int v) : value(v) {}

  friend constexpr auto operator+(WithOps l, WithOps r) { return WithOps{l.value + r.value}; }
  friend constexpr auto operator-(WithOps l, WithOps r) { return WithOps{l.value - r.value}; }
  friend constexpr auto operator*(WithOps l, WithOps r) { return WithOps{l.value * r.value}; }
  friend constexpr auto operator/(WithOps l, WithOps r) { return WithOps{l.value / r.value}; }
  friend constexpr auto operator%(WithOps l, WithOps r) { return WithOps{l.value % r.value}; }
  friend constexpr auto operator<<(WithOps l, WithOps r) { return WithOps{l.value << r.value}; }
  friend constexpr auto operator>>(WithOps l, WithOps r) { return WithOps{l.value >> r.value}; }
  friend constexpr auto operator&(WithOps l, WithOps r) { return WithOps{l.value & r.value}; }
  friend constexpr auto operator|(WithOps l, WithOps r) { return WithOps{l.value | r.value}; }
  friend constexpr auto operator^(WithOps l, WithOps r) { return WithOps{l.value ^ r.value}; }

  friend constexpr auto operator&&(WithOps l, WithOps r) { return WithOps{l.value && r.value}; }
  friend constexpr auto operator||(WithOps l, WithOps r) { return WithOps{l.value || r.value}; }
};

struct OpsReturnNonStructural {
  int value;

  constexpr OpsReturnNonStructural(int v) : value(v) {}

  friend constexpr auto operator+(OpsReturnNonStructural l, OpsReturnNonStructural r) {
    return NonStructural{l.value + r.value};
  }
  friend constexpr auto operator-(OpsReturnNonStructural l, OpsReturnNonStructural r) {
    return NonStructural{l.value - r.value};
  }
  friend constexpr auto operator*(OpsReturnNonStructural l, OpsReturnNonStructural r) {
    return NonStructural{l.value * r.value};
  }
  friend constexpr auto operator/(OpsReturnNonStructural l, OpsReturnNonStructural r) {
    return NonStructural{l.value / r.value};
  }
  friend constexpr auto operator%(OpsReturnNonStructural l, OpsReturnNonStructural r) {
    return NonStructural{l.value % r.value};
  }
  friend constexpr auto operator<<(OpsReturnNonStructural l, OpsReturnNonStructural r) {
    return NonStructural{l.value << r.value};
  }
  friend constexpr auto operator>>(OpsReturnNonStructural l, OpsReturnNonStructural r) {
    return NonStructural{l.value >> r.value};
  }
  friend constexpr auto operator&(OpsReturnNonStructural l, OpsReturnNonStructural r) {
    return NonStructural{l.value & r.value};
  }
  friend constexpr auto operator|(OpsReturnNonStructural l, OpsReturnNonStructural r) {
    return NonStructural{l.value | r.value};
  }
  friend constexpr auto operator^(OpsReturnNonStructural l, OpsReturnNonStructural r) {
    return NonStructural{l.value ^ r.value};
  }
  friend constexpr auto operator&&(OpsReturnNonStructural l, OpsReturnNonStructural r) {
    return NonStructural{l.value && r.value};
  }
  friend constexpr auto operator||(OpsReturnNonStructural l, OpsReturnNonStructural r) {
    return NonStructural{l.value || r.value};
  }
};

struct NoOps {};

template <class L, class R>
concept HasPlus = requires(L l, R r) {
  { l + r };
};

template <class L, class R>
concept HasMinus = requires(L l, R r) {
  { l - r };
};

template <class L, class R>
concept HasMultiply = requires(L l, R r) {
  { l * r };
};

template <class L, class R>
concept HasDivide = requires(L l, R r) {
  { l / r };
};

template <class L, class R>
concept HasModulo = requires(L l, R r) {
  { l % r };
};

template <class L, class R>
concept HasShiftLeft = requires(L l, R r) {
  { l << r };
};

template <class L, class R>
concept HasShiftRight = requires(L l, R r) {
  { l >> r };
};

template <class L, class R>
concept HasBitAnd = requires(L l, R r) {
  { l & r };
};

template <class L, class R>
concept HasBitOr = requires(L l, R r) {
  { l | r };
};

template <class L, class R>
concept HasBitXor = requires(L l, R r) {
  { l ^ r };
};

template <class L, class R>
concept HasLogicalAnd = requires(L l, R r) {
  { l && r };
};

template <class L, class R>
concept HasLogicalOr = requires(L l, R r) {
  { l || r };
};

template <class L, class R>
concept HasNoexceptPlus = requires(L l, R r) {
  { l + r } noexcept;
};

template <class L, class R>
concept HasNoexceptMinus = requires(L l, R r) {
  { l - r } noexcept;
};

template <class L, class R>
concept HasNoexceptMultiply = requires(L l, R r) {
  { l * r } noexcept;
};

template <class L, class R>
concept HasNoexceptDivide = requires(L l, R r) {
  { l / r } noexcept;
};

template <class L, class R>
concept HasNoexceptModulo = requires(L l, R r) {
  { l % r } noexcept;
};

template <class L, class R>
concept HasNoexceptShiftLeft = requires(L l, R r) {
  { l << r } noexcept;
};

template <class L, class R>
concept HasNoexceptShiftRight = requires(L l, R r) {
  { l >> r } noexcept;
};

template <class L, class R>
concept HasNoexceptBitAnd = requires(L l, R r) {
  { l & r } noexcept;
};

template <class L, class R>
concept HasNoexceptBitOr = requires(L l, R r) {
  { l | r } noexcept;
};

template <class L, class R>
concept HasNoexceptBitXor = requires(L l, R r) {
  { l ^ r } noexcept;
};

template <class L, class R>
concept HasNoexceptLogicalAnd = requires(L l, R r) {
  { l && r } noexcept;
};

template <class L, class R>
concept HasNoexceptLogicalOr = requires(L l, R r) {
  { l || r } noexcept;
};

// Concept checks for int + int operations
static_assert(HasPlus<std::constant_wrapper<6>, std::constant_wrapper<3>>);
static_assert(HasMinus<std::constant_wrapper<6>, std::constant_wrapper<3>>);
static_assert(HasMultiply<std::constant_wrapper<6>, std::constant_wrapper<3>>);
static_assert(HasDivide<std::constant_wrapper<6>, std::constant_wrapper<3>>);
static_assert(HasModulo<std::constant_wrapper<6>, std::constant_wrapper<3>>);
static_assert(HasShiftLeft<std::constant_wrapper<6>, std::constant_wrapper<1>>);
static_assert(HasShiftRight<std::constant_wrapper<6>, std::constant_wrapper<1>>);
static_assert(HasBitAnd<std::constant_wrapper<6>, std::constant_wrapper<3>>);
static_assert(HasBitOr<std::constant_wrapper<6>, std::constant_wrapper<3>>);
static_assert(HasBitXor<std::constant_wrapper<6>, std::constant_wrapper<3>>);
static_assert(HasLogicalAnd<std::constant_wrapper<6>, std::constant_wrapper<3>>);
static_assert(HasLogicalOr<std::constant_wrapper<6>, std::constant_wrapper<3>>);

static_assert(HasNoexceptPlus<std::constant_wrapper<6>, std::constant_wrapper<3>>);
static_assert(HasNoexceptMinus<std::constant_wrapper<6>, std::constant_wrapper<3>>);
static_assert(HasNoexceptMultiply<std::constant_wrapper<6>, std::constant_wrapper<3>>);
static_assert(HasNoexceptDivide<std::constant_wrapper<6>, std::constant_wrapper<3>>);
static_assert(HasNoexceptModulo<std::constant_wrapper<6>, std::constant_wrapper<3>>);
static_assert(HasNoexceptShiftLeft<std::constant_wrapper<6>, std::constant_wrapper<1>>);
static_assert(HasNoexceptShiftRight<std::constant_wrapper<6>, std::constant_wrapper<1>>);
static_assert(HasNoexceptBitAnd<std::constant_wrapper<6>, std::constant_wrapper<3>>);
static_assert(HasNoexceptBitOr<std::constant_wrapper<6>, std::constant_wrapper<3>>);
static_assert(HasNoexceptBitXor<std::constant_wrapper<6>, std::constant_wrapper<3>>);
static_assert(HasNoexceptLogicalAnd<std::constant_wrapper<6>, std::constant_wrapper<3>>);
static_assert(HasNoexceptLogicalOr<std::constant_wrapper<6>, std::constant_wrapper<3>>);

// NoOps
static_assert(!HasPlus<std::constant_wrapper<NoOps{}>, std::constant_wrapper<NoOps{}>>);
static_assert(!HasMinus<std::constant_wrapper<NoOps{}>, std::constant_wrapper<NoOps{}>>);
static_assert(!HasMultiply<std::constant_wrapper<NoOps{}>, std::constant_wrapper<NoOps{}>>);
static_assert(!HasDivide<std::constant_wrapper<NoOps{}>, std::constant_wrapper<NoOps{}>>);
static_assert(!HasModulo<std::constant_wrapper<NoOps{}>, std::constant_wrapper<NoOps{}>>);
static_assert(!HasShiftLeft<std::constant_wrapper<NoOps{}>, std::constant_wrapper<NoOps{}>>);
static_assert(!HasShiftRight<std::constant_wrapper<NoOps{}>, std::constant_wrapper<NoOps{}>>);
static_assert(!HasBitAnd<std::constant_wrapper<NoOps{}>, std::constant_wrapper<NoOps{}>>);
static_assert(!HasBitOr<std::constant_wrapper<NoOps{}>, std::constant_wrapper<NoOps{}>>);
static_assert(!HasBitXor<std::constant_wrapper<NoOps{}>, std::constant_wrapper<NoOps{}>>);
static_assert(!HasLogicalAnd<std::constant_wrapper<NoOps{}>, std::constant_wrapper<NoOps{}>>);
static_assert(!HasLogicalOr<std::constant_wrapper<NoOps{}>, std::constant_wrapper<NoOps{}>>);

// Concept checks for WithOps operations
static_assert(HasNoexceptPlus<std::constant_wrapper<WithOps{6}>, std::constant_wrapper<WithOps{3}>>);
static_assert(HasNoexceptMinus<std::constant_wrapper<WithOps{6}>, std::constant_wrapper<WithOps{3}>>);
static_assert(HasNoexceptMultiply<std::constant_wrapper<WithOps{6}>, std::constant_wrapper<WithOps{3}>>);
static_assert(HasNoexceptDivide<std::constant_wrapper<WithOps{6}>, std::constant_wrapper<WithOps{3}>>);
static_assert(HasNoexceptModulo<std::constant_wrapper<WithOps{6}>, std::constant_wrapper<WithOps{3}>>);
static_assert(HasNoexceptShiftLeft<std::constant_wrapper<WithOps{6}>, std::constant_wrapper<WithOps{1}>>);
static_assert(HasNoexceptShiftRight<std::constant_wrapper<WithOps{6}>, std::constant_wrapper<WithOps{1}>>);
static_assert(HasNoexceptBitAnd<std::constant_wrapper<WithOps{6}>, std::constant_wrapper<WithOps{3}>>);
static_assert(HasNoexceptBitOr<std::constant_wrapper<WithOps{6}>, std::constant_wrapper<WithOps{3}>>);
static_assert(HasNoexceptBitXor<std::constant_wrapper<WithOps{6}>, std::constant_wrapper<WithOps{3}>>);
static_assert(HasNoexceptLogicalAnd<std::constant_wrapper<WithOps{6}>, std::constant_wrapper<WithOps{3}>>);
static_assert(HasNoexceptLogicalOr<std::constant_wrapper<WithOps{6}>, std::constant_wrapper<WithOps{3}>>);

// clang-format off
// Non-structural types use implicit conversion to underlying type
static_assert(HasPlus<std::constant_wrapper<OpsReturnNonStructural{6}>, std::constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(HasMinus<std::constant_wrapper<OpsReturnNonStructural{6}>, std::constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(HasMultiply<std::constant_wrapper<OpsReturnNonStructural{6}>, std::constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(HasDivide<std::constant_wrapper<OpsReturnNonStructural{6}>, std::constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(HasModulo<std::constant_wrapper<OpsReturnNonStructural{6}>, std::constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(HasShiftLeft<std::constant_wrapper<OpsReturnNonStructural{6}>, std::constant_wrapper<OpsReturnNonStructural{1}>>);
static_assert(HasShiftRight<std::constant_wrapper<OpsReturnNonStructural{6}>, std::constant_wrapper<OpsReturnNonStructural{1}>>);
static_assert(HasBitAnd<std::constant_wrapper<OpsReturnNonStructural{6}>, std::constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(HasBitOr<std::constant_wrapper<OpsReturnNonStructural{6}>, std::constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(HasBitXor<std::constant_wrapper<OpsReturnNonStructural{6}>, std::constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(HasLogicalAnd<std::constant_wrapper<OpsReturnNonStructural{6}>, std::constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(HasLogicalOr<std::constant_wrapper<OpsReturnNonStructural{6}>, std::constant_wrapper<OpsReturnNonStructural{3}>>);

static_assert(!HasNoexceptPlus<std::constant_wrapper<OpsReturnNonStructural{6}>, std::constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasNoexceptMinus<std::constant_wrapper<OpsReturnNonStructural{6}>, std::constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasNoexceptMultiply<std::constant_wrapper<OpsReturnNonStructural{6}>, std::constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasNoexceptDivide<std::constant_wrapper<OpsReturnNonStructural{6}>, std::constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasNoexceptModulo<std::constant_wrapper<OpsReturnNonStructural{6}>, std::constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasNoexceptShiftLeft<std::constant_wrapper<OpsReturnNonStructural{6}>, std::constant_wrapper<OpsReturnNonStructural{1}>>);
static_assert(!HasNoexceptShiftRight<std::constant_wrapper<OpsReturnNonStructural{6}>, std::constant_wrapper<OpsReturnNonStructural{1}>>);
static_assert(!HasNoexceptBitAnd<std::constant_wrapper<OpsReturnNonStructural{6}>, std::constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasNoexceptBitOr<std::constant_wrapper<OpsReturnNonStructural{6}>, std::constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasNoexceptBitXor<std::constant_wrapper<OpsReturnNonStructural{6}>, std::constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasNoexceptLogicalAnd<std::constant_wrapper<OpsReturnNonStructural{6}>, std::constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasNoexceptLogicalOr<std::constant_wrapper<OpsReturnNonStructural{6}>, std::constant_wrapper<OpsReturnNonStructural{3}>>);
// clang-format on

constexpr bool test() {
  {
    // int + int
    std::constant_wrapper<6> cw6;
    std::constant_wrapper<3> cw3;

    std::same_as<std::constant_wrapper<9>> decltype(auto) result = cw6 + cw3;
    static_assert(result == 9);

    std::same_as<std::constant_wrapper<3>> decltype(auto) result2 = cw6 - cw3;
    static_assert(result2 == 3);

    std::same_as<std::constant_wrapper<18>> decltype(auto) result3 = cw6 * cw3;
    static_assert(result3 == 18);

    std::same_as<std::constant_wrapper<2>> decltype(auto) result4 = cw6 / cw3;
    static_assert(result4 == 2);

    std::same_as<std::constant_wrapper<0>> decltype(auto) result5 = cw6 % cw3;
    static_assert(result5 == 0);

    std::same_as<std::constant_wrapper<2>> decltype(auto) result6 = cw6 & cw3;
    static_assert(result6 == 2);

    std::same_as<std::constant_wrapper<7>> decltype(auto) result7 = cw6 | cw3;
    static_assert(result7 == 7);

    std::same_as<std::constant_wrapper<5>> decltype(auto) result8 = cw6 ^ cw3;
    static_assert(result8 == 5);

    // Shift operations: 6 << 3 = 48, 6 >> 3 = 0
    std::same_as<std::constant_wrapper<48>> decltype(auto) result9 = cw6 << cw3;
    static_assert(result9 == 48);

    std::same_as<std::constant_wrapper<0>> decltype(auto) result10 = cw6 >> cw3;
    static_assert(result10 == 0);

    // logical operations: int convertible to bool, so constant_wrapper overload is disabled
    // They are implicitly converted to bool and use built-in operators, resulting in a bool
    std::same_as<bool> decltype(auto) result11 = cw6 && cw3;
    assert(result11 == true);

    std::constant_wrapper<0> cw0;
    std::same_as<bool> decltype(auto) result12 = cw0 || cw3;
    assert(result12 == true);
  }

  {
    // WithOps operations
    std::constant_wrapper<WithOps{6}> cwWithOps6;
    std::constant_wrapper<WithOps{3}> cwWithOps3;

    std::same_as<std::constant_wrapper<WithOps{9}>> decltype(auto) result = cwWithOps6 + cwWithOps3;
    static_assert(result.value.value == 9);

    std::same_as<std::constant_wrapper<WithOps{3}>> decltype(auto) result2 = cwWithOps6 - cwWithOps3;
    static_assert(result2.value.value == 3);

    std::same_as<std::constant_wrapper<WithOps{18}>> decltype(auto) result3 = cwWithOps6 * cwWithOps3;
    static_assert(result3.value.value == 18);

    std::same_as<std::constant_wrapper<WithOps{2}>> decltype(auto) result4 = cwWithOps6 / cwWithOps3;
    static_assert(result4.value.value == 2);

    std::same_as<std::constant_wrapper<WithOps{0}>> decltype(auto) result5 = cwWithOps6 % cwWithOps3;
    static_assert(result5.value.value == 0);

    std::same_as<std::constant_wrapper<WithOps{2}>> decltype(auto) result6 = cwWithOps6 & cwWithOps3;
    static_assert(result6.value.value == 2);

    std::same_as<std::constant_wrapper<WithOps{7}>> decltype(auto) result7 = cwWithOps6 | cwWithOps3;
    static_assert(result7.value.value == 7);

    std::same_as<std::constant_wrapper<WithOps{5}>> decltype(auto) result8 = cwWithOps6 ^ cwWithOps3;
    static_assert(result8.value.value == 5);

    // Shift operations: 6 << 3 = 48, 6 >> 3 = 0
    std::same_as<std::constant_wrapper<WithOps{48}>> decltype(auto) result9 = cwWithOps6 << cwWithOps3;
    static_assert(result9.value.value == 48);

    std::same_as<std::constant_wrapper<WithOps{0}>> decltype(auto) result10 = cwWithOps6 >> cwWithOps3;
    static_assert(result10.value.value == 0);

    std::same_as<std::constant_wrapper<WithOps{1}>> decltype(auto) result11 = cwWithOps6 && cwWithOps3;
    static_assert(result11.value.value == 1);

    std::same_as<std::constant_wrapper<WithOps{1}>> decltype(auto) result12 = cwWithOps6 || cwWithOps3;
    static_assert(result12.value.value == 1);
  }

  {
    // Non-structural return types use implicit conversion
    std::constant_wrapper<OpsReturnNonStructural{6}> cwOpt6;
    std::constant_wrapper<OpsReturnNonStructural{3}> cwOpt3;

    std::same_as<NonStructural> decltype(auto) result = cwOpt6 + cwOpt3;
    assert(result.get() == 9);

    std::same_as<NonStructural> decltype(auto) result2 = cwOpt6 - cwOpt3;
    assert(result2.get() == 3);

    std::same_as<NonStructural> decltype(auto) result3 = cwOpt6 * cwOpt3;
    assert(result3.get() == 18);

    std::same_as<NonStructural> decltype(auto) result4 = cwOpt6 / cwOpt3;
    assert(result4.get() == 2);

    std::same_as<NonStructural> decltype(auto) result5 = cwOpt6 % cwOpt3;
    assert(result5.get() == 0);

    std::same_as<NonStructural> decltype(auto) result6 = cwOpt6 & cwOpt3;
    assert(result6.get() == 2);

    std::same_as<NonStructural> decltype(auto) result7 = cwOpt6 | cwOpt3;
    assert(result7.get() == 7);

    std::same_as<NonStructural> decltype(auto) result8 = cwOpt6 ^ cwOpt3;
    assert(result8.get() == 5);

    // Shift operations: 6 << 3 = 48, 6 >> 3 = 0
    std::same_as<NonStructural> decltype(auto) result9 = cwOpt6 << cwOpt3;
    assert(result9.get() == 48);

    std::same_as<NonStructural> decltype(auto) result10 = cwOpt6 >> cwOpt3;
    assert(result10.get() == 0);

    std::same_as<NonStructural> decltype(auto) result11 = cwOpt6 && cwOpt3;
    assert(result11.get() == 1);

    std::same_as<NonStructural> decltype(auto) result12 = cwOpt6 || cwOpt3;
    assert(result12.get() == 1);
  }

  {
    // Mix with runtime param: these operators are not used
    std::constant_wrapper<6> cw6;
    int i = 3;

    std::same_as<int> decltype(auto) result = cw6 + i;
    assert(result == 9);

    std::same_as<int> decltype(auto) result2 = cw6 - i;
    assert(result2 == 3);

    std::same_as<int> decltype(auto) result3 = cw6 * i;
    assert(result3 == 18);

    std::same_as<int> decltype(auto) result4 = cw6 / i;
    assert(result4 == 2);

    std::same_as<int> decltype(auto) result5 = cw6 % i;
    assert(result5 == 0);

    std::same_as<int> decltype(auto) result6 = cw6 & i;
    assert(result6 == 2);

    std::same_as<int> decltype(auto) result7 = cw6 | i;
    assert(result7 == 7);

    std::same_as<int> decltype(auto) result8 = cw6 ^ i;
    assert(result8 == 5);

    // Shift operations: 6 << 3 = 48, 6 >> 3 = 0
    std::same_as<int> decltype(auto) result9 = cw6 << i;
    assert(result9 == 48);

    std::same_as<int> decltype(auto) result10 = cw6 >> i;
    assert(result10 == 0);

    std::same_as<bool> decltype(auto) result11 = cw6 && i;
    assert(result11 == true);

    std::constant_wrapper<0> cw0;
    std::same_as<bool> decltype(auto) result12 = cw0 || i;
    assert(result12 == true);
  }

  {
    // with integral_constant
    std::constant_wrapper<6> cw6;
    std::integral_constant<int, 3> ic3;

    std::same_as<std::constant_wrapper<9>> decltype(auto) result = cw6 + ic3;
    static_assert(result == 9);

    std::same_as<std::constant_wrapper<3>> decltype(auto) result2 = cw6 - ic3;
    static_assert(result2 == 3);

    std::same_as<std::constant_wrapper<18>> decltype(auto) result3 = cw6 * ic3;
    static_assert(result3 == 18);

    std::same_as<std::constant_wrapper<2>> decltype(auto) result4 = cw6 / ic3;
    static_assert(result4 == 2);

    std::same_as<std::constant_wrapper<0>> decltype(auto) result5 = cw6 % ic3;
    static_assert(result5 == 0);

    std::same_as<std::constant_wrapper<2>> decltype(auto) result6 = cw6 & ic3;
    static_assert(result6 == 2);

    std::same_as<std::constant_wrapper<7>> decltype(auto) result7 = cw6 | ic3;
    static_assert(result7 == 7);

    std::same_as<std::constant_wrapper<5>> decltype(auto) result8 = cw6 ^ ic3;
    static_assert(result8 == 5);

    // Shift operations: 6 << 3 = 48, 6 >> 3 = 0
    std::same_as<std::constant_wrapper<48>> decltype(auto) result9 = cw6 << ic3;
    static_assert(result9 == 48);

    std::same_as<std::constant_wrapper<0>> decltype(auto) result10 = cw6 >> ic3;
    static_assert(result10 == 0);

    // logical operations: int convertible to bool, so constant_wrapper overload is disabled
    // They are implicitly converted to bool and use built-in operators, resulting in a bool
    std::same_as<bool> decltype(auto) result11 = cw6 && ic3;
    assert(result11 == true);

    std::constant_wrapper<0> cw0;
    std::same_as<bool> decltype(auto) result12 = cw0 || ic3;
    assert(result12 == true);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
