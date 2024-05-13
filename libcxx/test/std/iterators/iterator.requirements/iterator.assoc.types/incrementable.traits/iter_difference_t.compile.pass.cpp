//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<class T>
// using iter_difference_t;

#include <iterator>

#include <concepts>
#include <vector>

template <class T>
constexpr bool has_no_iter_difference_t() {
  return !requires { typename std::iter_difference_t<T>; };
}

template <class T, class Expected>
constexpr bool check_iter_difference_t() {
  constexpr bool result = std::same_as<std::iter_difference_t<T>, Expected>;
  static_assert(std::same_as<std::iter_difference_t<T const>, Expected> == result);
  static_assert(std::same_as<std::iter_difference_t<T volatile>, Expected> == result);
  static_assert(std::same_as<std::iter_difference_t<T const volatile>, Expected> == result);
  static_assert(std::same_as<std::iter_difference_t<T const&>, Expected> == result);
  static_assert(std::same_as<std::iter_difference_t<T volatile&>, Expected> == result);
  static_assert(std::same_as<std::iter_difference_t<T const volatile&>, Expected> == result);
  static_assert(std::same_as<std::iter_difference_t<T const&&>, Expected> == result);
  static_assert(std::same_as<std::iter_difference_t<T volatile&&>, Expected> == result);
  static_assert(std::same_as<std::iter_difference_t<T const volatile&&>, Expected> == result);

  return result;
}

static_assert(check_iter_difference_t<int, int>());
static_assert(check_iter_difference_t<int*, std::ptrdiff_t>());
static_assert(check_iter_difference_t<std::vector<int>::iterator, std::ptrdiff_t>());

struct int_subtraction {
  friend int operator-(int_subtraction, int_subtraction);
};
static_assert(check_iter_difference_t<int_subtraction, int>());

static_assert(has_no_iter_difference_t<void>());
static_assert(has_no_iter_difference_t<double>());

struct S {};
static_assert(has_no_iter_difference_t<S>());

struct void_subtraction {
  friend void operator-(void_subtraction, void_subtraction);
};
static_assert(has_no_iter_difference_t<void_subtraction>());
