//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<input_iterator I, sentinel_for<I> S, class T,
//          indirectly-binary-left-foldable<T, I> F>
//   constexpr see below ranges::fold_left_with_iter(I first, S last, T init, F f);
//
// template<input_range R, class T, indirectly-binary-left-foldable<T, iterator_t<R>> F>
//   constexpr see below ranges::fold_left_with_iter(R&& r, T init, F f);

// template<input_iterator I, sentinel_for<I> S, class T,
//          indirectly-binary-left-foldable<T, I> F>
//   constexpr see below ranges::fold_left(I first, S last, T init, F f);
//
// template<input_range R, class T, indirectly-binary-left-foldable<T, iterator_t<R>> F>
//   constexpr see below ranges::fold_left(R&& r, T init, F f);

// Checks that the algorithm requirements reject parameters that don't meet the overloads' constraints.

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <functional>
#include <iterator>
#include <ranges>

#include "test_iterators.h"

// FIXME(cjdb): deduplicate
struct bad_iterator_category {
  using value_type        = int;
  using difference_type   = std::ptrdiff_t;
  using iterator_category = void;

  value_type operator*() const;

  bad_iterator_category& operator++();
  void operator++(int);
};

// Covers indirectly_readable<I> too
template <std::input_or_output_iterator T>
  requires(!std::input_iterator<T>)
void requires_input_iterator() {
  struct bad_range {
    T begin();
    std::unreachable_sentinel_t end();
  };

  static_assert(!requires(bad_range r) {
    std::ranges::fold_left_with_iter(r.begin(), r.end(), std::unreachable_sentinel, 0, std::plus());
  });
  static_assert(!requires(bad_range r) { std::ranges::fold_left_with_iter(r, 0, std::plus()); });

  static_assert(!requires(bad_range r) {
    std::ranges::fold_left(r.begin(), r.end(), std::unreachable_sentinel, 0, std::plus());
  });

  static_assert(!requires(bad_range r) { std::ranges::fold_left(r, 0, std::plus()); });
}

template <std::equality_comparable S>
  requires(!std::sentinel_for<int*, S>)
void requires_sentinel() {
  static_assert(!requires(S first, S last) { std::ranges::fold_left_with_iter(first, last, 0, std::plus()); });
  static_assert(!requires(S first, S last) { std::ranges::fold_left(first, last, 0, std::plus()); });
}

struct non_copy_constructible_callable {
  non_copy_constructible_callable(non_copy_constructible_callable&&)      = default;
  non_copy_constructible_callable(non_copy_constructible_callable const&) = delete;

  int operator()(int, int) const;
};

template <class F>
  requires(!std::copy_constructible<F>)
void requires_copy_constructible_F() {
  static_assert(!requires(std::ranges::subrange<int*, int*> r, F f) {
    std::ranges::fold_left_with_iter(r.begin(), r.end(), 0, std::move(f));
  });
  static_assert(!requires(std::ranges::subrange<int*, int*> r, F f) {
    std::ranges::fold_left_with_iter(r, 0, std::move(f));
  });

  static_assert(!requires(std::ranges::subrange<int*, int*> r, F f) {
    std::ranges::fold_left(r.begin(), r.end(), 0, std::move(f));
  });
  static_assert(!requires(std::ranges::subrange<int*, int*> r, F f) { std::ranges::fold_left(r, 0, std::move(f)); });
}

struct not_invocable_with_lvalue_rhs {
  int operator()(int, int&&);
};

template <class F>
  requires(!std::invocable<F&, int, std::iter_reference_t<int*>>)
void requires_raw_invocable() {
  static_assert(!requires(std::ranges::subrange<int*, int*> r, F f) {
    std::ranges::fold_left_with_iter(r.begin(), r.end(), 0, f);
  });
  static_assert(!requires(std::ranges::subrange<int*, int*> r, F f) { std::ranges::fold_left_with_iter(r, 0, f); });

  static_assert(!requires(std::ranges::subrange<int*, int*> r, F f) {
    std::ranges::fold_left(r.begin(), r.end(), 0, f);
  });
  static_assert(!requires(std::ranges::subrange<int*, int*> r, F f) { std::ranges::fold_left(r, 0, f); });
}

struct S {};

struct non_decayable_result {
  S volatile& operator()(S, S) const;
};

template <std::invocable<S, std::iter_reference_t<S*>> F>
  requires(!std::convertible_to<std::invoke_result_t<F&, S, std::iter_reference_t<S*>>,
                                std::decay_t<std::invoke_result_t<F&, S, std::iter_reference_t<S*>>>>)
void requires_decaying_invoke_result() {
  static_assert(!requires(std::ranges::subrange<S*, S*> r, S init, F f) {
    std::ranges::fold_left_with_iter(r.begin(), r.end(), init, f);
  });
  static_assert(!requires(std::ranges::subrange<S*, S*> r, S init, F f) {
    std::ranges::fold_left_with_iter(r, init, f);
  });

  static_assert(!requires(std::ranges::subrange<S*, S*> r, S init, F f) {
    std::ranges::fold_left(r.begin(), r.end(), init, f);
  });
  static_assert(!requires(std::ranges::subrange<S*, S*> r, S init, F f) { std::ranges::fold_left(r, init, f); });
}

struct non_movable {
  non_movable(int);
  non_movable(non_movable&&) = delete;

  int apply(non_movable const&) const;
};

template <class T>
  requires(!std::movable<T>)
void requires_movable_init() {
  static_assert(!requires(std::ranges::subrange<T*, T*> r, T init) {
    std::ranges::fold_left_with_iter(r.begin(), r.end(), init, &T::apply);
  });
  static_assert(!requires(std::ranges::subrange<T*, T*> r, T init) {
    std::ranges::fold_left_with_iter(r, init, &T::apply);
  });
  static_assert(!requires(std::ranges::subrange<T*, T*> r, T init) {
    std::ranges::fold_left(r.begin(), r.end(), init, &T::apply);
  });
  static_assert(!requires(std::ranges::subrange<T*, T*> r, T init) { std::ranges::fold_left(r, init, &T::apply); });
}

struct result_not_movable_after_decay {
  result_not_movable_after_decay(int);
  result_not_movable_after_decay(result_not_movable_after_decay&&) = delete;
  result_not_movable_after_decay(result_not_movable_after_decay const&);

  friend result_not_movable_after_decay const& operator+(int, result_not_movable_after_decay const&);
  friend result_not_movable_after_decay const& operator+(result_not_movable_after_decay const&, int);
  friend result_not_movable_after_decay const&
  operator+(result_not_movable_after_decay const&, result_not_movable_after_decay const&);
};

template <class T>
  requires(!std::movable<T>)
void requires_movable_decayed() {
  static_assert(!requires(std::ranges::subrange<T*, T*> r) {
    std::ranges::fold_left_with_iter(r.begin(), r.end(), 0, std::plus());
  });
  static_assert(!requires(std::ranges::subrange<T*, T*> r) { std::ranges::fold_left_with_iter(r, 0, std::plus()); });

  static_assert(!requires(std::ranges::subrange<T*, T*> r) {
    std::ranges::fold_left(r.begin(), r.end(), 0, T::apply);
  });
  static_assert(!requires(std::ranges::subrange<T*, T*> r) { std::ranges::fold_left(r, 0, std::plus()); });
}

struct not_convertible_to_int {
  friend int operator+(not_convertible_to_int, not_convertible_to_int);
  friend int operator+(not_convertible_to_int, int);
  friend int operator+(int, not_convertible_to_int);
};

template <class T>
  requires(!std::convertible_to<T, int>)
void requires_init_is_convertible_to_decayed() {
  static_assert(!requires(std::ranges::subrange<int*, int*> r, T init) {
    std::ranges::fold_left_with_iter(r.begin(), r.end(), init, std::plus());
  });
  static_assert(!requires(std::ranges::subrange<int*, int*> r, T init) {
    std::ranges::fold_left_with_iter(r, init, std::plus());
  });

  static_assert(!requires(std::ranges::subrange<int*, int*> r, T init) {
    std::ranges::fold_left(r.begin(), r.end(), init, std::plus());
  });
  static_assert(!requires(std::ranges::subrange<int*, int*> r, T init) {
    std::ranges::fold_left(r, init, std::plus());
  });
}

struct not_invocable_with_decayed {
  not_invocable_with_decayed(int);
  friend not_invocable_with_decayed& operator+(int, not_invocable_with_decayed&);
  friend not_invocable_with_decayed& operator+(not_invocable_with_decayed&, int);
  friend not_invocable_with_decayed& operator+(not_invocable_with_decayed volatile&, not_invocable_with_decayed&);
};

template <class T>
  requires(!std::invocable<std::plus<>&, T, T&>)
void requires_invocable_with_decayed() {
  static_assert(!requires(std::ranges::subrange<T*, T*> r, int init) {
    std::ranges::fold_left_with_iter(r.begin(), r.end(), init, std::plus());
  });
  static_assert(!requires(std::ranges::subrange<T*, T*> r, int init) {
    std::ranges::fold_left_with_iter(r, init, std::plus());
  });

  static_assert(!requires(std::ranges::subrange<T*, T*> r, int init) {
    std::ranges::fold_left(r.begin(), r.end(), init, std::plus());
  });
  static_assert(!requires(std::ranges::subrange<T*, T*> r, int init) { std::ranges::fold_left(r, init, std::plus()); });
}

struct not_assignable_to_decayed {
  not_assignable_to_decayed();
  not_assignable_to_decayed(not_assignable_to_decayed&);
  not_assignable_to_decayed(not_assignable_to_decayed const&);
  not_assignable_to_decayed(not_assignable_to_decayed volatile&);
  not_assignable_to_decayed(not_assignable_to_decayed const volatile&);
  friend not_assignable_to_decayed volatile& operator+(not_assignable_to_decayed, not_assignable_to_decayed);
};

template <class T>
  requires(!std::assignable_from<T&, T volatile&>)
void requires_assignable_from_invoke_result() {
  static_assert(!requires(std::ranges::subrange<T*, T*> r, T init) {
    std::ranges::fold_left_with_iter(r.begin(), r.end(), init, std::plus());
  });
  static_assert(!requires(std::ranges::subrange<T*, T*> r, T init) {
    std::ranges::fold_left_with_iter(r, init, std::plus());
  });

  static_assert(!requires(std::ranges::subrange<T*, T*> r, T init) {
    std::ranges::fold_left(r.begin(), r.end(), init, std::plus());
  });
  static_assert(!requires(std::ranges::subrange<T*, T*> r, T init) { std::ranges::fold_left(r, init, std::plus()); });
}

void test() {
  requires_input_iterator<bad_iterator_category>();
  requires_sentinel<cpp17_input_iterator<int*>>();
  requires_copy_constructible_F<non_copy_constructible_callable>();
  requires_raw_invocable<not_invocable_with_lvalue_rhs>();
  requires_decaying_invoke_result<non_decayable_result>();
  requires_movable_init<non_movable>();
  requires_movable_decayed<result_not_movable_after_decay>();
  requires_init_is_convertible_to_decayed<not_convertible_to_int>();
  requires_invocable_with_decayed<not_invocable_with_decayed>();
  requires_assignable_from_invoke_result<not_assignable_to_decayed>();
}
