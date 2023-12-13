//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// Checks that `std::ranges::fold_left`'s requirements are correct.

#include <algorithm>
#include <concepts>
#include <functional>
#include <iterator>
#include <ranges>

#include "test_iterators.h"
#include "../requirements.h"

// Covers indirectly_readable<I> too
template <std::input_or_output_iterator T>
  requires(!std::input_iterator<T>)
void requires_input_iterator() {
  static_assert(!requires(T t) { std::ranges::fold_left(t, std::unreachable_sentinel, 0, std::plus()); });
}

template <std::equality_comparable T>
  requires(!std::sentinel_for<int*, T>)
void requires_sentinel() {
  static_assert(!requires(T first, T last) { std::ranges::fold_left(first, last, 0, std::plus()); });
}

template <class F>
  requires(!std::copy_constructible<F>)
void requires_copy_constructible_F() {
  static_assert(!requires(int* first, int* last, F f) { std::ranges::fold_left(first, last, 0, std::move(f)); });
}

template <class F>
  requires(!std::invocable<F&, int, std::iter_reference_t<int*>>)
void requires_raw_invocable() {
  static_assert(!requires(int* first, int* last, F f) { std::ranges::fold_left(first, last, 0, f); });
}

template <class F>
  requires(!std::convertible_to<std::invoke_result_t<F&, S, std::iter_reference_t<S*>>,
                                std::decay_t<std::invoke_result_t<F&, S, std::iter_reference_t<S*>>>>)
void requires_decaying_invoke_result() {
  static_assert(!requires(S* first, S* last, S init, F f) { std::ranges::fold_left(first, last, init, f); });
}

template <class T>
  requires(!std::movable<T>)
void requires_movable_init() {
  static_assert(!requires(copyable_non_movable* first, copyable_non_movable* last, T init) {
    std::ranges::fold_left(first, last, init, std::plus());
  });
}

template <class T>
  requires(!std::movable<T>)
void requires_movable_decayed() {
  static_assert(!requires(T* first, T* last) { std::ranges::fold_left(first, last, 0, std::minus()); });
}

template <class T>
  requires(!std::convertible_to<T, int>)
void requires_init_is_convertible_to_decayed() {
  static_assert(!requires(int* first, int* last, T init) { std::ranges::fold_left(first, last, init, std::plus()); });
}

template <class T>
  requires(!std::invocable<std::plus<>&, T, T&>)
void requires_invocable_with_decayed() {
  static_assert(!requires(T* first, T* last, int init) { std::ranges::fold_left(first, last, init, std::plus()); });
}

template <class T>
  requires(!std::assignable_from<T&, T volatile&>)
void requires_assignable_from_invoke_result() {
  static_assert(!requires(T* first, T* last, T init) { std::ranges::fold_left(first, last, init, std::plus()); });
}

void test() {
  requires_input_iterator<bad_iterator_category>();
  requires_sentinel<cpp17_input_iterator<int*>>();
  requires_copy_constructible_F<non_copy_constructible_callable>();
  requires_raw_invocable<not_invocable>();
  requires_decaying_invoke_result<non_decayable_result>();
  requires_movable_init<non_movable>();
  requires_movable_decayed<copyable_non_movable>();
  requires_init_is_convertible_to_decayed<not_convertible_to_int>();
  requires_invocable_with_decayed<not_invocable_with_decayed>();
  requires_assignable_from_invoke_result<not_assignable_to_decayed>();
}
