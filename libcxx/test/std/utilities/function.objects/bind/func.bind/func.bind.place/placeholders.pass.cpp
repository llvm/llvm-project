//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>

// namespace placeholders {
//   // M is the implementation-defined number of placeholders
//   extern unspecified _1;
//   extern unspecified _2;
//   .
//   .
//   .
//   extern unspecified _Mp;
// }

// The Standard recommends implementing them as `inline constexpr` in C++17.
//
// Libc++ implements the placeholders as `extern const` in all standard modes
// to avoid an ABI break in C++03: making them `inline constexpr` requires removing
// their definition in the shared library to avoid ODR violations, which is an
// ABI break.
//
// Concretely, `extern const` is almost indistinguishable from constexpr for the
// placeholders since they are empty types.

#include <functional>
#include <type_traits>

#include "test_macros.h"

template <class T>
TEST_CONSTEXPR_CXX17 void test(const T& t) {
  // Test default constructible.
  {
    T x; (void)x;
  }

  // Test copy constructible.
  {
    T x = t; (void)x;
    static_assert(std::is_nothrow_copy_constructible<T>::value, "");
    static_assert(std::is_nothrow_move_constructible<T>::value, "");
  }

  // It is implementation-defined whether placeholder types are CopyAssignable.
  // CopyAssignable placeholders' copy assignment operators shall not throw exceptions.
#ifdef _LIBCPP_VERSION
  {
    T x;
    x = t;
    static_assert(std::is_nothrow_copy_assignable<T>::value, "");
    static_assert(std::is_nothrow_move_assignable<T>::value, "");
  }
#endif
}

TEST_CONSTEXPR_CXX17 bool test_all() {
  test(std::placeholders::_1);
  test(std::placeholders::_2);
  test(std::placeholders::_3);
  test(std::placeholders::_4);
  test(std::placeholders::_5);
  test(std::placeholders::_6);
  test(std::placeholders::_7);
  test(std::placeholders::_8);
  test(std::placeholders::_9);
  test(std::placeholders::_10);
  return true;
}

int main(int, char**) {
  test_all();
#if TEST_STD_VER >= 17
  static_assert(test_all(), "");
#endif

  return 0;
}
