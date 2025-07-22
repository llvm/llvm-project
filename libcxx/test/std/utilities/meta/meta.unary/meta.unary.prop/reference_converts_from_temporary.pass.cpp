//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// These compilers don't support std::reference_converts_from_temporary yet.
// UNSUPPORTED: android, apple-clang-16, clang-19.1

// <type_traits>

// template<class T, class U> struct reference_converts_from_temporary;

// template<class T, class U>
// constexpr bool reference_converts_from_temporary_v
//   = reference_converts_from_temporary<T, U>::value;

#include <cassert>
#include <type_traits>

#include "common.h"
#include "test_macros.h"

template <typename T, typename U, bool Expected>
constexpr void test_reference_converts_from_temporary() {
  assert((std::reference_converts_from_temporary<T, U>::value == Expected));
  assert((std::reference_converts_from_temporary_v<T, U> == Expected));
}

constexpr bool test() {
  test_reference_converts_from_temporary<int&, int&, false>();
  test_reference_converts_from_temporary<int&, int&, false>();
  test_reference_converts_from_temporary<int&, int&&, false>();

  test_reference_converts_from_temporary<const int&, int&, false>();
  test_reference_converts_from_temporary<const int&, const int&, false>();
  test_reference_converts_from_temporary<const int&, int&&, false>();

  test_reference_converts_from_temporary<int&, long&, false>(); // doesn't construct

  test_reference_converts_from_temporary<const int&, long&, true>();
  test_reference_converts_from_temporary<const int&, long&&, true>();
  test_reference_converts_from_temporary<int&&, long&, true>();

  assert((std::is_constructible_v<int&, ConvertsToRef<int, int&>>));
  test_reference_converts_from_temporary<int&, ConvertsToRef<int, int&>, false>();

  assert((std::is_constructible_v<int&&, ConvertsToRef<int, int&&>>));
  test_reference_converts_from_temporary<int&&, ConvertsToRef<int, int&&>, false>();

  assert((std::is_constructible_v<const int&, ConvertsToRef<int, const int&>>));
  test_reference_converts_from_temporary<int&&, ConvertsToRef<int, const int&>, false>();

  assert((std::is_constructible_v<const int&, ConvertsToRef<long, long&>>));
  test_reference_converts_from_temporary<const int&, ConvertsToRef<long, long&>, true>();
#ifndef TEST_COMPILER_GCC
  test_reference_converts_from_temporary<const int&, ConvertsToRefPrivate<long, long&>, false>();
#endif

  // Test that it doesn't accept non-reference types as input.
  test_reference_converts_from_temporary<int, long, false>();

  test_reference_converts_from_temporary<const int&, long, true>();

  // Additional checks
  test_reference_converts_from_temporary<const Base&, Derived, true>();
  test_reference_converts_from_temporary<int&&, int, true>();
  test_reference_converts_from_temporary<const int&, int, true>();
  test_reference_converts_from_temporary<int&&, int&&, false>();
  test_reference_converts_from_temporary<const int&, int&&, false>();
  test_reference_converts_from_temporary<int&&, long&&, true>();
  test_reference_converts_from_temporary<int&&, long, true>();

  test_reference_converts_from_temporary<int&, ExplicitConversionRef, false>();
  test_reference_converts_from_temporary<const int&, ExplicitConversionRef, true>();
  test_reference_converts_from_temporary<int&&, ExplicitConversionRvalueRef, true>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
