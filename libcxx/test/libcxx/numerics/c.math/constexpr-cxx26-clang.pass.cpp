//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check that Clang supports constexpr <cmath> functions mentioned in the
// P1383R2 paper that is part of C++26 (https://wg21.link/p1383r2)
//
// Every function called in this test should become constexpr. Whenever some
// of the desired functions become constexpr, the programmer switches
// `ASSERT_NOT_CONSTEXPR_CXX26` to `ASSERT_CONSTEXPR_CXX26`, and new functions
// are added as they get implemented, until the paper is fully implemented.
//
// REQUIRES: clang
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// We don't control the implementation of these functions on windows
// UNSUPPORTED: windows

// Missing some math functions.
// XFAIL: LLVM-LIBC-FIXME

#include <cmath>

#define ASSERT_CONSTEXPR_CXX26(Expr) static_assert(__builtin_constant_p(Expr) && (Expr))
#define ASSERT_NOT_CONSTEXPR_CXX26(Expr) static_assert(!__builtin_constant_p(Expr))

int main(int, char**) {
  ASSERT_CONSTEXPR_CXX26(std::exp(0.0f) == 1.0f);
  ASSERT_CONSTEXPR_CXX26(std::exp(0.0) == 1.0);
  ASSERT_CONSTEXPR_CXX26(std::exp(0) == 1.0);
  ASSERT_CONSTEXPR_CXX26(std::expf(0.0f) == 1.0f);

  return 0;
}
