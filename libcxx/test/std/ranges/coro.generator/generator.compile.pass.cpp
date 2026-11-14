//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <generator>

// template<class Ref, class Val = void, class Allocator = void>
//   class generator;

#include <generator>

#include <concepts>
#include <cstddef>
#include <iterator>
#include <memory>
#include <memory_resource>
#include <ranges>
#include <tuple>
#include <type_traits>

#include "MoveOnly.h"

template <class G, class V, class R, class RR>
constexpr bool conformance() {
  static_assert(std::ranges::range<G>);
  static_assert(std::ranges::view<G>);
  static_assert(std::ranges::input_range<G>);
  static_assert(!std::ranges::forward_range<G>);
  static_assert(!std::ranges::borrowed_range<G>);
  static_assert(!std::ranges::range<const G>);
  static_assert(std::derived_from<G, std::ranges::view_interface<G>>);

  static_assert(!std::is_default_constructible_v<G>);
  static_assert(!std::is_copy_constructible_v<G>);
  static_assert(!std::is_copy_assignable_v<G>);
  static_assert(std::is_nothrow_move_constructible_v<G>);
  static_assert(std::is_nothrow_move_assignable_v<G>);
  static_assert(std::is_nothrow_destructible_v<G>);

  static_assert(std::same_as<std::ranges::range_value_t<G>, V>);
  static_assert(std::same_as<std::ranges::range_reference_t<G>, R>);
  static_assert(std::same_as<std::ranges::range_rvalue_reference_t<G>, RR>);
  static_assert(std::same_as<std::ranges::range_difference_t<G>, std::ptrdiff_t>);
  static_assert(std::same_as<std::ranges::sentinel_t<G>, std::default_sentinel_t>);
  using Yielded = std::conditional_t<std::is_reference_v<R>, R, const R&>;
  static_assert(std::same_as<typename G::yielded, Yielded>);

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

static_assert(conformance<std::generator<MoveOnly>, MoveOnly, MoveOnly&&, MoveOnly&&>());
static_assert(conformance<std::generator<MoveOnly&>, MoveOnly, MoveOnly&, MoveOnly&&>());
static_assert(conformance<std::generator<const MoveOnly&>, MoveOnly, const MoveOnly&, const MoveOnly&&>());

// Reference and value need not be the same type. Exercise valid common-reference relationships.
template <class T>
struct Proxy {
  Proxy(const T&);
};

static_assert(conformance<std::generator<Proxy<int>, int>, int, Proxy<int>, Proxy<int>>());
static_assert(conformance<std::generator<Proxy<int>&, int>, int, Proxy<int>&, Proxy<int>&&>());
static_assert(conformance<std::generator<const Proxy<int>&, int>, int, const Proxy<int>&, const Proxy<int>&&>());
static_assert(conformance<std::generator<Proxy<int>&&, int>, int, Proxy<int>&&, Proxy<int>&&>());
static_assert(conformance<std::generator<const Proxy<int>&&, int>, int, const Proxy<int>&&, const Proxy<int>&&>());

using Reference = std::tuple<int&, int&>;
using Value     = std::tuple<int, int>;
static_assert(conformance<std::generator<Reference, Value>, Value, Reference, Reference>());

static_assert(std::same_as<std::generator<int>, std::generator<int, void, void>>);
static_assert(std::same_as<std::pmr::generator<int>, std::generator<int, void, std::pmr::polymorphic_allocator<>>>);
static_assert(
    std::same_as<std::pmr::generator<int&, int>, std::generator<int&, int, std::pmr::polymorphic_allocator<>>>);
static_assert(conformance<std::generator<int, void, std::allocator<std::byte>>, int, int&&, int&&>());
static_assert(conformance<std::pmr::generator<int>, int, int&&, int&&>());
