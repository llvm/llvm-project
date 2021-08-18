//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// template<class T>
// using iter_value_t;

#include <iterator>

#include <concepts>
#include <memory>
#include <vector>

template <class T, class Expected>
constexpr bool check_iter_value_t() {
  constexpr bool result = std::same_as<std::iter_value_t<T>, Expected>;
  static_assert(std::same_as<std::iter_value_t<T const>, Expected> == result);
  static_assert(std::same_as<std::iter_value_t<T volatile>, Expected> == result);
  static_assert(std::same_as<std::iter_value_t<T const volatile>, Expected> == result);
  static_assert(std::same_as<std::iter_value_t<T const&>, Expected> == result);
  static_assert(std::same_as<std::iter_value_t<T volatile&>, Expected> == result);
  static_assert(std::same_as<std::iter_value_t<T const volatile&>, Expected> == result);
  static_assert(std::same_as<std::iter_value_t<T const&&>, Expected> == result);
  static_assert(std::same_as<std::iter_value_t<T volatile&&>, Expected> == result);
  static_assert(std::same_as<std::iter_value_t<T const volatile&&>, Expected> == result);

  return result;
}

static_assert(check_iter_value_t<int*, int>());
static_assert(check_iter_value_t<int[], int>());
static_assert(check_iter_value_t<int[10], int>());
static_assert(check_iter_value_t<std::vector<int>::iterator, int>());
static_assert(check_iter_value_t<std::shared_ptr<int>, int>());

struct both_members {
  using value_type = double;
  using element_type = double;
};
static_assert(check_iter_value_t<both_members, double>());

// clang-format off
template <class T>
requires requires { typename std::iter_value_t<T>; }
constexpr bool check_no_iter_value_t() {
  return false;
}
// clang-format on

template <class T>
constexpr bool check_no_iter_value_t() {
  return true;
}

static_assert(check_no_iter_value_t<void>());
static_assert(check_no_iter_value_t<double>());

struct S {};
static_assert(check_no_iter_value_t<S>());

struct different_value_element_members {
  using value_type = int;
  using element_type = long;
};
static_assert(check_no_iter_value_t<different_value_element_members>());

int main(int, char**) { return 0; }
