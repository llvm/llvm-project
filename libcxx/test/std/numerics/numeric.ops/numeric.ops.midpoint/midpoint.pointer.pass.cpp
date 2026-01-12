//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// MSVC warning C5215: a function parameter with a volatile qualified type is deprecated in C++20
// MSVC warning C5216: a volatile qualified return type is deprecated in C++20
// ADDITIONAL_COMPILE_FLAGS(cl-style-warnings): /wd5215 /wd5216

// <numeric>

// template <class _Tp>
// _Tp* midpoint(_Tp* __a, _Tp* __b) noexcept
// Constraints:
//  - T is a complete object type.

#include <cassert>
#include <cstddef>
#include <numeric>
#include <type_traits>
#include <utility>

#include "test_macros.h"

using FuncPtr = void (*)();
struct Incomplete;

template <typename T>
concept has_midpoint = requires(T a, T b) { std::midpoint(a, b); };

static_assert(!has_midpoint<std::nullptr_t>);
static_assert(!has_midpoint<FuncPtr>);
LIBCPP_STATIC_ASSERT(!has_midpoint<Incomplete*>);

static_assert(!has_midpoint<void*>);
static_assert(!has_midpoint<const void*>);
static_assert(!has_midpoint<volatile void*>);
static_assert(!has_midpoint<const volatile void*>);

template <typename T>
constexpr bool check(T* base, std::ptrdiff_t i, std::ptrdiff_t j, std::ptrdiff_t expect) {
  return std::midpoint(base + i, base + j) == base + expect;
}

template <typename T>
constexpr bool test_pointer() {
  ASSERT_SAME_TYPE(decltype(std::midpoint(std::declval<T*>(), std::declval<T*>())), T*);
  ASSERT_NOEXCEPT(std::midpoint(std::declval<T*>(), std::declval<T*>()));

  std::remove_cv_t<T> array[20] = {};
  assert(check(array, 0, 0, 0));
  assert(check(array, 1, 1, 1));
  assert(check(array, 0, 9, 4));
  assert(check(array, 0, 10, 5));
  assert(check(array, 0, 11, 5));
  assert(check(array, 9, 0, 5));
  assert(check(array, 10, 0, 5));
  assert(check(array, 11, 0, 6));
  assert(check(array, 0, 18, 9));
  assert(check(array, 2, 12, 7));

  return true;
}

template <typename T>
void test() {
  assert(test_pointer<T>());
  assert(test_pointer<const T>());
  assert(test_pointer<volatile T>());
  assert(test_pointer<const volatile T>());

  static_assert(test_pointer<T>());
  static_assert(test_pointer<const T>());
}

int main(int, char**) {
  test<char>();
  test<int>();
  test<double>();

  return 0;
}
