//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <type_traits>

// template<class T, class U> struct reference_constructs_from_temporary;

// template<class T, class U>
// constexpr bool reference_constructs_from_temporary_v
//   = reference_constructs_from_temporary<T, U>::value;

#include <cassert>
#include <type_traits>

#include "common.h"
#include "test_macros.h"

template <typename T, typename U, bool Expected>
constexpr void test_reference_constructs_from_temporary() {
  assert((std::reference_constructs_from_temporary<T, U>::value == Expected));
  assert((std::reference_constructs_from_temporary_v<T, U> == Expected));
}

constexpr bool test() {
  test_reference_constructs_from_temporary<int&, int&, false>();
  test_reference_constructs_from_temporary<int&, int&, false>();
  test_reference_constructs_from_temporary<int&, int&&, false>();

  test_reference_constructs_from_temporary<const int&, int&, false>();
  test_reference_constructs_from_temporary<const int&, const int&, false>();
  test_reference_constructs_from_temporary<const int&, int&&, false>();

  test_reference_constructs_from_temporary<int&, long&, false>(); // doesn't construct

  test_reference_constructs_from_temporary<const int&, long&, true>();
  test_reference_constructs_from_temporary<const int&, long&&, true>();
  test_reference_constructs_from_temporary<int&&, long&, true>();

  assert((std::is_constructible_v<int&, ConvertsToRef<int, int&>>));
  test_reference_constructs_from_temporary<int&, ConvertsToRef<int, int&>, false>();

  assert((std::is_constructible_v<int&&, ConvertsToRef<int, int&&>>));
  test_reference_constructs_from_temporary<int&&, ConvertsToRef<int, int&&>, false>();

  assert((std::is_constructible_v<const int&, ConvertsToRef<int, const int&>>));
  test_reference_constructs_from_temporary<int&&, ConvertsToRef<int, const int&>, false>();

  assert((std::is_constructible_v<const int&, ConvertsToRef<long, long&>>));
  test_reference_constructs_from_temporary<const int&, ConvertsToRef<long, long&>, true>();
#ifndef TEST_COMPILER_GCC
  // TODO: Remove this guard once https://gcc.gnu.org/bugzilla/show_bug.cgi?id=120529 gets fixed.
  test_reference_constructs_from_temporary<const int&, ConvertsToRefPrivate<long, long&>, false>();
#endif

  // Test that it doesn't accept non-reference types as input.
  test_reference_constructs_from_temporary<int, long, false>();

  test_reference_constructs_from_temporary<const int&, long, true>();

#if defined(TEST_COMPILER_GCC) ||                                                                                      \
    (defined(TEST_CLANG_VER) &&                                                                                        \
     ((!defined(__ANDROID__) && TEST_CLANG_VER >= 2100) || (defined(__ANDROID__) && TEST_CLANG_VER >= 2200))) ||       \
    (defined(TEST_APPLE_CLANG_VER) && TEST_APPLE_CLANG_VER >= 1800)
  // TODO: Bump the version numbers if newer Apple Clang or Android Clang hasn't implemented LWG3819 yet.
  // TODO: Remove this guard once no supported Clang is affected by https://llvm.org/PR114344.

  // Test function references.
  test_reference_constructs_from_temporary<void (&)(), void(), false>();
  test_reference_constructs_from_temporary<void (&&)(), void(), false>();

  // Test cv-qualification dropping for scalar prvalues. LWG3819 also covers this.
  test_reference_constructs_from_temporary<int&&, const int, true>();
  test_reference_constructs_from_temporary<int&&, volatile int, true>();
#endif

#if !defined(TEST_COMPILER_GCC) &&                                                                                     \
    ((defined(TEST_CLANG_VER) &&                                                                                       \
      ((!defined(__ANDROID__) && TEST_CLANG_VER >= 2100) || (defined(__ANDROID__) && TEST_CLANG_VER >= 2200))) ||      \
     (defined(TEST_APPLE_CLANG_VER) && TEST_APPLE_CLANG_VER >= 1800))
  // TODO: Bump the version numbers if newer Apple Clang or Android Clang hasn't implemented LWG3819 yet.
  // TODO: Remove this guard once supported Clang and GCC have LWG3819 implemented.

  // Test LWG3819: reference_meows_from_temporary should not use is_meowible.
  test_reference_constructs_from_temporary<ConvertsFromNonMovable&&, NonMovable, true>();
#endif

  // Additional checks
  test_reference_constructs_from_temporary<const Base&, Derived, true>();
  test_reference_constructs_from_temporary<int&&, int, true>();
  test_reference_constructs_from_temporary<const int&, int, true>();
  test_reference_constructs_from_temporary<int&&, int&&, false>();
  test_reference_constructs_from_temporary<const int&, int&&, false>();
  test_reference_constructs_from_temporary<int&&, long&&, true>();
  test_reference_constructs_from_temporary<int&&, long, true>();

  test_reference_constructs_from_temporary<int&, ExplicitConversionRef, false>();
  test_reference_constructs_from_temporary<const int&, ExplicitConversionRef, false>();
  test_reference_constructs_from_temporary<int&&, ExplicitConversionRvalueRef, false>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
