//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: gcc-10

// template<class T>
// using iter_difference_t;

#include <iterator>

#include <concepts>
#include <vector>

template <class T, class Expected>
[[nodiscard]] constexpr bool check_iter_difference_t() {
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
  friend int operator-(int_subtraction, int_subtraction) noexcept;
};
static_assert(check_iter_difference_t<int_subtraction, int>());

// clang-format off
template <class T>
requires requires { typename std::iter_difference_t<T>; }
[[nodiscard]] constexpr bool check_no_iter_difference_t() {
  return false;
}
// clang-format on

template <class T>
[[nodiscard]] constexpr bool check_no_iter_difference_t() {
  return true;
}

static_assert(check_no_iter_difference_t<void>());
static_assert(check_no_iter_difference_t<double>());

struct S {};
static_assert(check_no_iter_difference_t<S>());

struct void_subtraction {
  friend void operator-(void_subtraction, void_subtraction);
};
static_assert(check_no_iter_difference_t<void_subtraction>());

int main(int, char**) { return 0; }
