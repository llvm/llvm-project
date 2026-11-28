//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <generator>
// [coro.generator.iterator]

#include <generator>

#include <cassert>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <memory>
#include <ranges>
#include <type_traits>
#include <utility>

template <class Ref, class Val = void>
std::generator<Ref, Val> generate(int& value) {
  using R = std::conditional_t<std::is_void_v<Val>, Ref&&, Ref>;
  co_yield static_cast<R>(value);
  ++value;
  co_yield static_cast<R>(value);
}

template <class Ref, class Val = void>
void test() {
  using G = std::generator<Ref, Val>;
  using I = std::ranges::iterator_t<G>;
  using R = std::conditional_t<std::is_void_v<Val>, Ref&&, Ref>;
  static_assert(std::input_iterator<I>);
  static_assert(!std::forward_iterator<I>);
  static_assert(!std::is_copy_constructible_v<I>);
  static_assert(!std::is_copy_assignable_v<I>);
  static_assert(std::is_nothrow_move_constructible_v<I>);
  static_assert(std::is_nothrow_move_assignable_v<I>);
  static_assert(std::same_as<typename I::value_type, int>);
  static_assert(std::same_as<typename I::difference_type, std::ptrdiff_t>);
  static_assert(std::same_as<decltype(*std::declval<const I&>()), R>);
  static_assert(std::same_as<decltype(std::declval<I&>()++), void>);

  int value     = 1;
  auto g        = generate<Ref, Val>(value);
  auto i        = g.begin();
  auto j        = std::move(i);
  const auto& c = j;
  // LWG 3762: comparison must accept a const lvalue of this move-only iterator.
  static_assert(noexcept(c == std::default_sentinel));
  assert(!(c == std::default_sentinel));
  assert(!(std::default_sentinel == c));
  assert(c != std::default_sentinel);
  assert(std::default_sentinel != c);
  assert(*c == 1);
  if constexpr (std::is_reference_v<R>) {
    auto&& reference = *c;
    assert(std::addressof(reference) == std::addressof(value));
  }

  std::same_as<I&> decltype(auto) assigned = (i = std::move(j));
  assert(&assigned == &i);
  std::same_as<I&> decltype(auto) incremented = ++i;
  assert(&incremented == &i);
  assert(*i == 2);
  i++;
  assert(i == std::default_sentinel);
  assert(std::default_sentinel == i);
  assert(!(i != std::default_sentinel));
  assert(!(std::default_sentinel != i));
}

int main(int, char**) {
  test<int>();
  test<int, int>();
  test<int&>();
  test<const int&>();
  test<int&&>();
  test<const int&&>();
  return 0;
}
