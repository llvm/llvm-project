//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <generator>

// template<class Ref, class V = void, class Allocator = void>
//   class generator;

#include <generator>

#include <cassert>
#include <concepts>
#include <ranges>
#include <type_traits>

template <class G, class V, class R, class RR>
constexpr bool conformance() {
  static_assert(std::ranges::range<G>);
  static_assert(std::ranges::view<G>);
  static_assert(std::ranges::input_range<G>);
  static_assert(!std::ranges::forward_range<G>);
  static_assert(!std::ranges::borrowed_range<G>);

  static_assert(std::same_as<std::ranges::range_value_t<G>, V>);
  static_assert(std::same_as<std::ranges::range_reference_t<G>, R>);
  static_assert(std::same_as<std::ranges::range_rvalue_reference_t<G>, RR>);

  return true;
}

static_assert(conformance<std::generator<int>, int, int&&, int&&>());
static_assert(conformance<std::generator<int, int>, int, int, int>());

static_assert(conformance<std::generator<int&>, int, int&, int&&>());
static_assert(conformance<std::generator<int&, int>, int, int&, int&&>());
static_assert(conformance<std::generator<const int&>, int, const int&, const int&&>());
static_assert(conformance<std::generator<const int&, int>, int, const int&, const int&&>());

static_assert(conformance<std::generator<int&&>, int, int&&, int&&>());
static_assert(conformance<std::generator<int&&, int>, int, int&&, int&&>());
static_assert(conformance<std::generator<const int&&>, int, const int&&, const int&&>());
static_assert(conformance<std::generator<const int&&, int>, int, const int&&, const int&&>());
