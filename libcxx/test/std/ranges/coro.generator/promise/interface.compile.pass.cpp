//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <generator>
// [coro.generator.promise]

#include <generator>

#include <concepts>
#include <coroutine>
#include <memory>
#include <memory_resource>
#include <utility>

#include "MoveOnly.h"

template <class P>
concept HasAwaitTransform = requires(P& p) { p.await_transform(); };

template <class G>
constexpr bool test() {
  using P = typename G::promise_type;
  using Y = typename G::yielded;
  static_assert(std::same_as<decltype(std::declval<P&>().get_return_object()), G>);
  static_assert(noexcept(std::declval<P&>().get_return_object()));
  static_assert(std::same_as<decltype(std::declval<const P&>().initial_suspend()), std::suspend_always>);
  static_assert(noexcept(std::declval<const P&>().initial_suspend()));
  static_assert(noexcept(std::declval<P&>().final_suspend()));
  static_assert(std::same_as<decltype(std::declval<P&>().yield_value(std::declval<Y>())), std::suspend_always>);
  static_assert(noexcept(std::declval<P&>().yield_value(std::declval<Y>())));
  static_assert(std::same_as<decltype(std::declval<const P&>().return_void()), void>);
  static_assert(noexcept(std::declval<const P&>().return_void()));
  static_assert(std::same_as<decltype(std::declval<P&>().unhandled_exception()), void>);
  static_assert(!HasAwaitTransform<P>);
  return true;
}

static_assert(test<std::generator<int>>());
static_assert(test<std::generator<int, int>>());
static_assert(test<std::generator<int&>>());
static_assert(test<std::generator<const int&>>());
static_assert(test<std::generator<int&&>>());
static_assert(test<std::generator<const int&&>>());
static_assert(test<std::generator<MoveOnly>>());
static_assert(test<std::generator<int, void, std::allocator<int>>>());
static_assert(test<std::pmr::generator<int>>());
