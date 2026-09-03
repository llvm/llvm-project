//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// codecvt<char32_t, char8_t, mbstate_t>
//  deprecated in C++20, per LWG3767

// REQUIRES: std-at-least-c++20
// XFAIL: availability-char8_t_support-missing

#include <locale>

#include "../with_public_dtor.hpp"

void test() {
  // Don't test for the exact type since the underlying type of std::mbstate_t depends on implementation details.

  // expected-warning-re@+1 {{'codecvt<char32_t, char8_t, {{.*}}>' is deprecated}}
  [[maybe_unused]] with_public_dtor<std::codecvt<char32_t, char8_t, std::mbstate_t>> cvt("", 0);
}
