//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// These compilers do not support __builtin_type_order
// UNSUPPORTED: clang-21, clang-22, clang-23, apple-clang-21

// <compare>

// template<class T, class U>
//   struct type_order;
// template<class T, class U>
//   constexpr strong_ordering type_order_v = type_order<T, U>::value;

#include <compare>
#include <concepts>

template <class T, class U>
constexpr bool test_order(std::strong_ordering expected) {
  return std::type_order<T, U>()() == expected && std::type_order<T, U>::value == expected &&
         std::type_order_v<T, U> == expected && static_cast<std::strong_ordering>(std::type_order<T, U>()) == expected;
}

template <class T, class U>
constexpr bool eq = test_order<T, U>(std::strong_ordering::equal) && test_order<U, T>(std::strong_ordering::equal);
template <class T, class U>
constexpr bool lt = test_order<T, U>(std::strong_ordering::less) && test_order<U, T>(std::strong_ordering::greater);
template <class T, class U>
constexpr bool gt = test_order<T, U>(std::strong_ordering::greater) && test_order<U, T>(std::strong_ordering::less);
template <class T, class U>
constexpr bool ne = (lt<T, U> || gt<T, U>);

struct A {};
struct B {};
struct C {};

static_assert(std::same_as<std::type_order<A, A>::value_type, std::strong_ordering>);
static_assert(std::same_as<decltype(std::type_order<A, A>::value), const std::strong_ordering>);
static_assert(std::same_as<decltype(std::type_order_v<A, A>), const std::strong_ordering>);

static_assert(noexcept(std::type_order<int, int>()()));
static_assert(noexcept(static_cast<std::strong_ordering>(std::type_order<int, int>())));

static_assert(ne<int, const int>);
static_assert(ne<int, int&>);

static_assert(eq<A, A>);
static_assert(ne<A, B>);

static_assert(lt<A, B> != gt<A, B>);
static_assert((!lt<A, B> || !lt<B, C> || lt<A, C>) && (!gt<A, B> || !gt<B, C> || gt<A, C>));

struct incomplete;
static_assert(eq<incomplete, incomplete>);
static_assert(ne<incomplete, A>);

template <auto>
constexpr bool test_template_arg = true;
static_assert(test_template_arg<std::type_order<A, incomplete>{}>);
