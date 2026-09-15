//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// constant_wrapper

// template<constexpr-param L, constexpr-param R>
//   friend constexpr auto operator<=>(L, R) noexcept
//     -> constant_wrapper<(L::value <=> R::value)>
//       { return {}; }
// template<constexpr-param L, constexpr-param R>
//   friend constexpr auto operator<(L, R) noexcept -> constant_wrapper<(L::value < R::value)>
//     { return {}; }
// template<constexpr-param L, constexpr-param R>
//   friend constexpr auto operator<=(L, R) noexcept -> constant_wrapper<(L::value <= R::value)>
//     { return {}; }
// template<constexpr-param L, constexpr-param R>
//   friend constexpr auto operator==(L, R) noexcept -> constant_wrapper<(L::value == R::value)>
//     { return {}; }
// template<constexpr-param L, constexpr-param R>
//   friend constexpr auto operator!=(L, R) noexcept -> constant_wrapper<(L::value != R::value)>
//     { return {}; }
// template<constexpr-param L, constexpr-param R>
//   friend constexpr auto operator>(L, R) noexcept -> constant_wrapper<(L::value > R::value)>
//     { return {}; }
// template<constexpr-param L, constexpr-param R>
//   friend constexpr auto operator>=(L, R) noexcept -> constant_wrapper<(L::value >= R::value)>
//     { return {}; }

#include <cassert>
#include <concepts>
#include <utility>

#include "helpers.h"

struct WithOps {
  int value;

  constexpr WithOps(int v) : value(v) {}

  friend constexpr auto operator==(WithOps l, WithOps r) { return l.value == r.value; }
  friend constexpr auto operator!=(WithOps l, WithOps r) { return l.value != r.value; }
  friend constexpr auto operator<(WithOps l, WithOps r) { return l.value < r.value; }
  friend constexpr auto operator<=(WithOps l, WithOps r) { return l.value <= r.value; }
  friend constexpr auto operator>=(WithOps l, WithOps r) { return l.value >= r.value; }
  friend constexpr auto operator>(WithOps l, WithOps r) { return l.value > r.value; }
  friend constexpr auto operator<=>(WithOps l, WithOps r) { return l.value <=> r.value; }
};

struct OpsReturnNonStructural {
  int value;

  constexpr OpsReturnNonStructural(int v) : value(v) {}

  friend constexpr auto operator==(OpsReturnNonStructural l, OpsReturnNonStructural r) {
    return NonStructural{l.value == r.value ? 1 : 0};
  }
  friend constexpr auto operator!=(OpsReturnNonStructural l, OpsReturnNonStructural r) {
    return NonStructural{l.value != r.value ? 1 : 0};
  }
  friend constexpr auto operator<(OpsReturnNonStructural l, OpsReturnNonStructural r) {
    return NonStructural{l.value < r.value ? 1 : 0};
  }
  friend constexpr auto operator<=(OpsReturnNonStructural l, OpsReturnNonStructural r) {
    return NonStructural{l.value <= r.value ? 1 : 0};
  }
  friend constexpr auto operator>=(OpsReturnNonStructural l, OpsReturnNonStructural r) {
    return NonStructural{l.value >= r.value ? 1 : 0};
  }
  friend constexpr auto operator>(OpsReturnNonStructural l, OpsReturnNonStructural r) {
    return NonStructural{l.value > r.value ? 1 : 0};
  }
  friend constexpr auto operator<=>(OpsReturnNonStructural l, OpsReturnNonStructural r) {
    return NonStructural{(l.value < r.value) ? -1 : (l.value > r.value) ? 1 : 0};
  }
};

struct NoOps {};

template <class L, class R>
concept HasEqual = requires(L l, R r) {
  { l == r };
};

template <class L, class R>
concept HasNotEqual = requires(L l, R r) {
  { l != r };
};

template <class L, class R>
concept HasLess = requires(L l, R r) {
  { l < r };
};

template <class L, class R>
concept HasLessEqual = requires(L l, R r) {
  { l <= r };
};

template <class L, class R>
concept HasGreater = requires(L l, R r) {
  { l > r };
};

template <class L, class R>
concept HasGreaterEqual = requires(L l, R r) {
  { l >= r };
};

template <class L, class R>
concept HasSpaceship = requires(L l, R r) {
  { l <=> r };
};

template <class L, class R>
concept HasNoexceptEqual = requires(L l, R r) {
  { l == r } noexcept;
};

template <class L, class R>
concept HasNoexceptNotEqual = requires(L l, R r) {
  { l != r } noexcept;
};

template <class L, class R>
concept HasNoexceptLess = requires(L l, R r) {
  { l < r } noexcept;
};

template <class L, class R>
concept HasNoexceptLessEqual = requires(L l, R r) {
  { l <= r } noexcept;
};

template <class L, class R>
concept HasNoexceptGreater = requires(L l, R r) {
  { l > r } noexcept;
};

template <class L, class R>
concept HasNoexceptGreaterEqual = requires(L l, R r) {
  { l >= r } noexcept;
};

template <class L, class R>
concept HasNoexceptSpaceship = requires(L l, R r) {
  { l <=> r } noexcept;
};

// Concept checks for int comparisons
static_assert(HasEqual<std::constant_wrapper<6>, std::constant_wrapper<3>>);
static_assert(HasNotEqual<std::constant_wrapper<6>, std::constant_wrapper<3>>);
static_assert(HasLess<std::constant_wrapper<6>, std::constant_wrapper<3>>);
static_assert(HasLessEqual<std::constant_wrapper<6>, std::constant_wrapper<3>>);
static_assert(HasGreater<std::constant_wrapper<6>, std::constant_wrapper<3>>);
static_assert(HasGreaterEqual<std::constant_wrapper<6>, std::constant_wrapper<3>>);
static_assert(HasSpaceship<std::constant_wrapper<6>, std::constant_wrapper<3>>);

static_assert(HasNoexceptEqual<std::constant_wrapper<6>, std::constant_wrapper<3>>);
static_assert(HasNoexceptNotEqual<std::constant_wrapper<6>, std::constant_wrapper<3>>);
static_assert(HasNoexceptLess<std::constant_wrapper<6>, std::constant_wrapper<3>>);
static_assert(HasNoexceptLessEqual<std::constant_wrapper<6>, std::constant_wrapper<3>>);
static_assert(HasNoexceptGreater<std::constant_wrapper<6>, std::constant_wrapper<3>>);
static_assert(HasNoexceptGreaterEqual<std::constant_wrapper<6>, std::constant_wrapper<3>>);
static_assert(HasNoexceptSpaceship<std::constant_wrapper<6>, std::constant_wrapper<3>>);

// NoOps
static_assert(!HasEqual<std::constant_wrapper<NoOps{}>, std::constant_wrapper<NoOps{}>>);
static_assert(!HasNotEqual<std::constant_wrapper<NoOps{}>, std::constant_wrapper<NoOps{}>>);
static_assert(!HasLess<std::constant_wrapper<NoOps{}>, std::constant_wrapper<NoOps{}>>);
static_assert(!HasLessEqual<std::constant_wrapper<NoOps{}>, std::constant_wrapper<NoOps{}>>);
static_assert(!HasGreater<std::constant_wrapper<NoOps{}>, std::constant_wrapper<NoOps{}>>);
static_assert(!HasGreaterEqual<std::constant_wrapper<NoOps{}>, std::constant_wrapper<NoOps{}>>);
static_assert(!HasSpaceship<std::constant_wrapper<NoOps{}>, std::constant_wrapper<NoOps{}>>);

// Concept checks for WithOps comparisons
static_assert(HasNoexceptEqual<std::constant_wrapper<WithOps{6}>, std::constant_wrapper<WithOps{3}>>);
static_assert(HasNoexceptNotEqual<std::constant_wrapper<WithOps{6}>, std::constant_wrapper<WithOps{3}>>);
static_assert(HasNoexceptLess<std::constant_wrapper<WithOps{6}>, std::constant_wrapper<WithOps{3}>>);
static_assert(HasNoexceptLessEqual<std::constant_wrapper<WithOps{6}>, std::constant_wrapper<WithOps{3}>>);
static_assert(HasNoexceptGreater<std::constant_wrapper<WithOps{6}>, std::constant_wrapper<WithOps{3}>>);
static_assert(HasNoexceptGreaterEqual<std::constant_wrapper<WithOps{6}>, std::constant_wrapper<WithOps{3}>>);
static_assert(!HasNoexceptSpaceship<std::constant_wrapper<WithOps{6}>, std::constant_wrapper<WithOps{3}>>,
              "strong_ordering is not a structural type, so the call falls back to runtime implicit conversion and "
              "operator<=>, which is noexcept(false)");

// clang-format off
// Non-structural types use implicit conversion to underlying type
static_assert(HasEqual<std::constant_wrapper<OpsReturnNonStructural{6}>, std::constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(HasNotEqual<std::constant_wrapper<OpsReturnNonStructural{6}>, std::constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(HasLess<std::constant_wrapper<OpsReturnNonStructural{6}>, std::constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(HasLessEqual<std::constant_wrapper<OpsReturnNonStructural{6}>, std::constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(HasGreater<std::constant_wrapper<OpsReturnNonStructural{6}>, std::constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(HasGreaterEqual<std::constant_wrapper<OpsReturnNonStructural{6}>, std::constant_wrapper<OpsReturnNonStructural{3}>>);

static_assert(!HasNoexceptEqual<std::constant_wrapper<OpsReturnNonStructural{6}>, std::constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasNoexceptNotEqual<std::constant_wrapper<OpsReturnNonStructural{6}>, std::constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasNoexceptLess<std::constant_wrapper<OpsReturnNonStructural{6}>, std::constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasNoexceptLessEqual<std::constant_wrapper<OpsReturnNonStructural{6}>, std::constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasNoexceptGreater<std::constant_wrapper<OpsReturnNonStructural{6}>, std::constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasNoexceptGreaterEqual<std::constant_wrapper<OpsReturnNonStructural{6}>, std::constant_wrapper<OpsReturnNonStructural{3}>>);
// clang-format on

constexpr bool test() {
  {
    // int comparisons: 6 vs 3 - returns constant_wrapper<bool_value>
    std::constant_wrapper<6> cw6;
    std::constant_wrapper<3> cw3;

    std::same_as<std::constant_wrapper<false>> decltype(auto) equal = cw6 == cw3;
    static_assert(!static_cast<bool>(equal));

    std::same_as<std::constant_wrapper<true>> decltype(auto) not_equal = cw6 != cw3;
    static_assert(static_cast<bool>(not_equal));

    std::same_as<std::constant_wrapper<false>> decltype(auto) less = cw6 < cw3;
    static_assert(!static_cast<bool>(less));

    std::same_as<std::constant_wrapper<false>> decltype(auto) less_equal = cw6 <= cw3;
    static_assert(!static_cast<bool>(less_equal));

    std::same_as<std::constant_wrapper<true>> decltype(auto) greater = cw6 > cw3;
    static_assert(static_cast<bool>(greater));

    std::same_as<std::constant_wrapper<true>> decltype(auto) greater_equal = cw6 >= cw3;
    static_assert(static_cast<bool>(greater_equal));

    // strong_ordering is not a structural type
    std::same_as<std::strong_ordering> decltype(auto) spaceship = cw6 <=> cw3;
    assert(spaceship == std::strong_ordering::greater);
  }

  {
    // int comparisons: equal values
    std::constant_wrapper<3> cw3a;
    std::constant_wrapper<3> cw3b;

    std::same_as<std::constant_wrapper<true>> decltype(auto) equal = cw3a == cw3b;
    static_assert(static_cast<bool>(equal));

    std::same_as<std::constant_wrapper<false>> decltype(auto) not_equal = cw3a != cw3b;
    static_assert(!static_cast<bool>(not_equal));

    std::same_as<std::constant_wrapper<false>> decltype(auto) less = cw3a < cw3b;
    static_assert(!static_cast<bool>(less));

    std::same_as<std::constant_wrapper<true>> decltype(auto) less_equal = cw3a <= cw3b;
    static_assert(static_cast<bool>(less_equal));

    std::same_as<std::constant_wrapper<true>> decltype(auto) greater = cw3a >= cw3b;
    static_assert(static_cast<bool>(greater));

    std::same_as<std::constant_wrapper<false>> decltype(auto) greater_cmp = cw3a > cw3b;
    static_assert(!static_cast<bool>(greater_cmp));

    std::same_as<std::strong_ordering> decltype(auto) spaceship = cw3a <=> cw3b;
    assert(spaceship == std::strong_ordering::equal);
  }

  {
    // WithOps comparisons - returns constant_wrapper<bool_value>
    std::constant_wrapper<WithOps{6}> cwWithOps6;
    std::constant_wrapper<WithOps{3}> cwWithOps3;

    std::same_as<std::constant_wrapper<false>> decltype(auto) equal = cwWithOps6 == cwWithOps3;
    static_assert(!static_cast<bool>(equal));

    std::same_as<std::constant_wrapper<true>> decltype(auto) not_equal = cwWithOps6 != cwWithOps3;
    static_assert(static_cast<bool>(not_equal));

    std::same_as<std::constant_wrapper<false>> decltype(auto) less = cwWithOps6 < cwWithOps3;
    static_assert(!static_cast<bool>(less));

    std::same_as<std::constant_wrapper<false>> decltype(auto) less_equal = cwWithOps6 <= cwWithOps3;
    static_assert(!static_cast<bool>(less_equal));

    std::same_as<std::constant_wrapper<true>> decltype(auto) greater = cwWithOps6 > cwWithOps3;
    static_assert(static_cast<bool>(greater));

    std::same_as<std::constant_wrapper<true>> decltype(auto) greater_equal = cwWithOps6 >= cwWithOps3;
    static_assert(static_cast<bool>(greater_equal));

    std::same_as<std::strong_ordering> decltype(auto) spaceship = cwWithOps6 <=> cwWithOps3;
    assert(spaceship == std::strong_ordering::greater);
  }

  {
    // WithOps comparisons: equal values
    std::constant_wrapper<WithOps{3}> cwWithOps3a;
    std::constant_wrapper<WithOps{3}> cwWithOps3b;

    std::same_as<std::constant_wrapper<true>> decltype(auto) equal = cwWithOps3a == cwWithOps3b;
    static_assert(static_cast<bool>(equal));

    std::same_as<std::constant_wrapper<false>> decltype(auto) not_equal = cwWithOps3a != cwWithOps3b;
    static_assert(!static_cast<bool>(not_equal));

    std::same_as<std::constant_wrapper<false>> decltype(auto) less = cwWithOps3a < cwWithOps3b;
    static_assert(!static_cast<bool>(less));

    std::same_as<std::constant_wrapper<true>> decltype(auto) less_equal = cwWithOps3a <= cwWithOps3b;
    static_assert(static_cast<bool>(less_equal));

    std::same_as<std::constant_wrapper<true>> decltype(auto) greater_equal = cwWithOps3a >= cwWithOps3b;
    static_assert(static_cast<bool>(greater_equal));

    std::same_as<std::constant_wrapper<false>> decltype(auto) greater = cwWithOps3a > cwWithOps3b;
    static_assert(!static_cast<bool>(greater));

    std::same_as<std::strong_ordering> decltype(auto) spaceship = cwWithOps3a <=> cwWithOps3b;
    assert(spaceship == std::strong_ordering::equal);
  }

  {
    // Non-structural return types use implicit conversion
    std::constant_wrapper<OpsReturnNonStructural{6}> cwOpt6;
    std::constant_wrapper<OpsReturnNonStructural{3}> cwOpt3;

    std::same_as<NonStructural> decltype(auto) equal = cwOpt6 == cwOpt3;
    assert(equal.get() == 0);

    std::same_as<NonStructural> decltype(auto) not_equal = cwOpt6 != cwOpt3;
    assert(not_equal.get() == 1);

    std::same_as<NonStructural> decltype(auto) less = cwOpt6 < cwOpt3;
    assert(less.get() == 0);

    std::same_as<NonStructural> decltype(auto) less_equal = cwOpt6 <= cwOpt3;
    assert(less_equal.get() == 0);

    std::same_as<NonStructural> decltype(auto) greater = cwOpt6 > cwOpt3;
    assert(greater.get() == 1);

    std::same_as<NonStructural> decltype(auto) greater_equal = cwOpt6 >= cwOpt3;
    assert(greater_equal.get() == 1);

    std::same_as<NonStructural> decltype(auto) spaceship = cwOpt6 <=> cwOpt3;
    assert(spaceship.get() == 1);
  }

  {
    // Mix with runtime param: these operators are not used (built-in operators)
    std::constant_wrapper<6> cw6;
    int i = 3;

    std::same_as<bool> decltype(auto) equal = cw6 == i;
    assert(!equal);

    std::same_as<bool> decltype(auto) not_equal = cw6 != i;
    assert(not_equal);

    std::same_as<bool> decltype(auto) less = cw6 < i;
    assert(!less);

    std::same_as<bool> decltype(auto) less_equal = cw6 <= i;
    assert(!less_equal);

    std::same_as<bool> decltype(auto) greater = cw6 > i;
    assert(greater);

    std::same_as<bool> decltype(auto) greater_equal = cw6 >= i;
    assert(greater_equal);

    std::same_as<std::strong_ordering> decltype(auto) spaceship = cw6 <=> i;
    assert(spaceship == std::strong_ordering::greater);
  }

  {
    // with integral_constant
    std::constant_wrapper<6> cw6;
    std::integral_constant<int, 3> ic3;

    std::same_as<std::constant_wrapper<false>> decltype(auto) equal = cw6 == ic3;
    static_assert(!static_cast<bool>(equal));

    std::same_as<std::constant_wrapper<true>> decltype(auto) not_equal = cw6 != ic3;
    static_assert(static_cast<bool>(not_equal));

    std::same_as<std::constant_wrapper<false>> decltype(auto) less = cw6 < ic3;
    static_assert(!static_cast<bool>(less));

    std::same_as<std::constant_wrapper<false>> decltype(auto) less_equal = cw6 <= ic3;
    static_assert(!static_cast<bool>(less_equal));

    std::same_as<std::constant_wrapper<true>> decltype(auto) greater = cw6 > ic3;
    static_assert(static_cast<bool>(greater));

    std::same_as<std::constant_wrapper<true>> decltype(auto) greater_equal = cw6 >= ic3;
    static_assert(static_cast<bool>(greater_equal));

    // strong_ordering is not a structural type
    std::same_as<std::strong_ordering> decltype(auto) spaceship = cw6 <=> ic3;
    assert(spaceship == std::strong_ordering::greater);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
