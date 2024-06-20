//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// std::ranges::range_adaptor_closure;

#include <ranges>

#include <algorithm>
#include <vector>

#include "test_range.h"

template <class T>
concept CanDeriveFromRangeAdaptorClosure = requires { typename std::ranges::range_adaptor_closure<T>; };
static_assert(!CanDeriveFromRangeAdaptorClosure<int>);

struct Foo {};
static_assert(CanDeriveFromRangeAdaptorClosure<Foo>);
static_assert(!CanDeriveFromRangeAdaptorClosure<Foo&>);
static_assert(!CanDeriveFromRangeAdaptorClosure<const Foo>);
static_assert(!CanDeriveFromRangeAdaptorClosure<volatile Foo>);
static_assert(!CanDeriveFromRangeAdaptorClosure<const volatile Foo&&>);

struct incomplete_t;
static_assert(CanDeriveFromRangeAdaptorClosure<incomplete_t>);

using range_t = std::vector<int>;

template <class T>
concept RangeAdaptorClosure =
    CanBePiped<range_t, T&> && CanBePiped<range_t, const T&> && CanBePiped<range_t, T&&> &&
    CanBePiped<range_t, const T&&>;

struct callable : std::ranges::range_adaptor_closure<callable> {
  static void operator()(const range_t&) {}
};
static_assert(RangeAdaptorClosure<callable>);

// `not_callable_1` doesn't have an `operator()`
struct not_callable_1 : std::ranges::range_adaptor_closure<not_callable_1> {};
static_assert(!RangeAdaptorClosure<not_callable_1>);

// `not_callable_2` doesn't have an `operator()` that accepts a `range` argument
struct not_callable_2 : std::ranges::range_adaptor_closure<not_callable_2> {
  static void operator()() {}
};
static_assert(!RangeAdaptorClosure<not_callable_2>);

// `not_derived_from_1` doesn't derive from `std::ranges::range_adaptor_closure`
struct not_derived_from_1 {
  static void operator()(const range_t&) {}
};
static_assert(!RangeAdaptorClosure<not_derived_from_1>);

// `not_derived_from_2` doesn't publicly derive from `std::ranges::range_adaptor_closure`
struct not_derived_from_2 : private std::ranges::range_adaptor_closure<not_derived_from_2> {
  static void operator()(const range_t&) {}
};
static_assert(!RangeAdaptorClosure<not_derived_from_2>);

// `not_derived_from_3` doesn't derive from the correct specialization of `std::ranges::range_adaptor_closure`
struct not_derived_from_3 : std::ranges::range_adaptor_closure<callable> {
  static void operator()(const range_t&) {}
};
static_assert(!RangeAdaptorClosure<not_derived_from_3>);

// `not_derived_from_4` doesn't derive from exactly one specialization of `std::ranges::range_adaptor_closure`
struct not_derived_from_4
    : std::ranges::range_adaptor_closure<not_derived_from_4>,
      std::ranges::range_adaptor_closure<callable> {
  static void operator()(const range_t&) {}
};
static_assert(!RangeAdaptorClosure<not_derived_from_4>);

// `is_range` models `range`
struct is_range : std::ranges::range_adaptor_closure<is_range> {
  static void operator()(const range_t&) {}
  int* begin() const { return nullptr; }
  int* end() const { return nullptr; }
};
static_assert(std::ranges::range<is_range> && std::ranges::range<const is_range>);
static_assert(!RangeAdaptorClosure<is_range>);

// user-defined range adaptor closure object
struct negate_fn : std::ranges::range_adaptor_closure<negate_fn> {
  template <std::ranges::range Range>
  static constexpr decltype(auto) operator()(Range&& range) {
    return std::forward<Range>(range) | std::views::transform([](auto element) { return -element; });
  }
};
static_assert(RangeAdaptorClosure<negate_fn>);
constexpr auto negate = negate_fn{};

// user-defined range adaptor closure object
struct plus_1_fn : std::ranges::range_adaptor_closure<plus_1_fn> {
  template <std::ranges::range Range>
  static constexpr decltype(auto) operator()(Range&& range) {
    return std::forward<Range>(range) | std::views::transform([](auto element) { return element + 1; });
  }
};
static_assert(RangeAdaptorClosure<plus_1_fn>);
constexpr auto plus_1 = plus_1_fn{};

constexpr bool test() {
  const std::vector<int> n{1, 2, 3, 4, 5};
  const std::vector<int> n_negate{-1, -2, -3, -4, -5};

  assert(std::ranges::equal(n | negate, n_negate));
  assert(std::ranges::equal(negate(n), n_negate));

  assert(std::ranges::equal(n | negate | negate, n));
  assert(std::ranges::equal(n | (negate | negate), n));
  assert(std::ranges::equal((n | negate) | negate, n));
  assert(std::ranges::equal(negate(n) | negate, n));
  assert(std::ranges::equal(negate(n | negate), n));
  assert(std::ranges::equal((negate | negate)(n), n));
  assert(std::ranges::equal(negate(negate(n)), n));

  const std::vector<int> n_plus_1_negate{-2, -3, -4, -5, -6};
  assert(std::ranges::equal(n | plus_1 | negate, n_plus_1_negate));
  assert(std::ranges::equal(
      n | plus_1 | std::views::transform([](auto element) { return element; }) | negate, n_plus_1_negate));

  const std::vector<int> n_negate_plus_1{0, -1, -2, -3, -4};
  assert(std::ranges::equal(n | negate | plus_1, n_negate_plus_1));
  assert(std::ranges::equal(n | std::views::reverse | negate | plus_1 | std::views::reverse, n_negate_plus_1));
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
