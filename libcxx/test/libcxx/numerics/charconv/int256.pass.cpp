//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// Requires compiler-rt __int256 builtins (__udivoi3, __umodoi3) at runtime.
// These are not yet available in the system compiler-rt library.
// REQUIRES: int256-runtime

// Test to_chars support for __uint256_t / __int256_t (Tier 3)

#include <charconv>
#include <cstring>
#include <limits>

#include "test_macros.h"

#ifdef TEST_HAS_NO_INT256
int main(int, char**) { return 0; }
#else

int main(int, char**) {
  char buf[80]; // 78 digits max + sign + null

  // to_chars: small values that fit in 64-bit
  {
    __uint256_t val = 42;
    auto [ptr, ec]  = std::to_chars(buf, buf + sizeof(buf), val);
    *ptr            = '\0';
    if (ec != std::errc{} || std::strcmp(buf, "42") != 0)
      return 1;
  }

  // to_chars: value that fits in 128-bit but not 64-bit
  {
    __uint256_t val = (__uint256_t)1 << 64;
    auto [ptr, ec]  = std::to_chars(buf, buf + sizeof(buf), val);
    *ptr            = '\0';
    if (ec != std::errc{} || std::strcmp(buf, "18446744073709551616") != 0)
      return 2;
  }

  // to_chars: value > 128-bit
  {
    // 2^128 = 340282366920938463463374607431768211456
    __uint256_t val = (__uint256_t)1 << 128;
    auto [ptr, ec]  = std::to_chars(buf, buf + sizeof(buf), val);
    *ptr            = '\0';
    if (ec != std::errc{} || std::strcmp(buf, "340282366920938463463374607431768211456") != 0)
      return 3;
  }

  // to_chars: zero
  {
    __uint256_t val = 0;
    auto [ptr, ec]  = std::to_chars(buf, buf + sizeof(buf), val);
    *ptr            = '\0';
    if (ec != std::errc{} || std::strcmp(buf, "0") != 0)
      return 4;
  }

  // to_chars: signed negative
  {
    __int256_t val = -1;
    auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf), val);
    *ptr           = '\0';
    if (ec != std::errc{} || std::strcmp(buf, "-1") != 0)
      return 5;
  }

  // to_chars: buffer too small
  {
    __uint256_t val = (__uint256_t)1 << 128;
    char small[5];
    auto [ptr, ec] = std::to_chars(small, small + sizeof(small), val);
    if (ec != std::errc::value_too_large)
      return 6;
  }

  return 0;
}
#endif
