//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <array>

// template <size_t I, class T, size_t N> T& get(array<T, N>& a);

#include "test_macros.h"

// Prevent -Warray-bounds from issuing a diagnostic when testing with clang verify.
TEST_CLANG_DIAGNOSTIC_IGNORED("-Warray-bounds")
TEST_GCC_DIAGNOSTIC_IGNORED("-Warray-bounds")

#include <array>
#include <cassert>

void f() {
  typedef double T;
  typedef std::array<T, 3> C;
  C c = {1, 2, 3.5};
  std::get<3>(c) = 5.5; // expected-note {{requested here}}
  // expected-error-re@array:* {{static assertion failed{{( due to requirement '3U[L]{0,2} < 3U[L]{0,2}')?}}{{.*}}Index out of bounds in std::get<> (std::array)}}
}
