//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// Test std::byteswap with __int256_t and __uint256_t

#include <bit>
#include <cassert>

#include "test_macros.h"

#ifdef TEST_HAS_NO_INT256
int main(int, char**) { return 0; }
#else

constexpr __uint256_t
make256(unsigned long long h3, unsigned long long h2, unsigned long long h1, unsigned long long h0) {
  __uint256_t v = (__uint256_t)h3;
  v             = (v << 64) | (__uint256_t)h2;
  v             = (v << 64) | (__uint256_t)h1;
  v             = (v << 64) | (__uint256_t)h0;
  return v;
}

// Constexpr tests
static_assert(std::byteswap((__uint256_t)0) == (__uint256_t)0);
static_assert(std::byteswap(~(__uint256_t)0) == ~(__uint256_t)0);

// Known pattern: bytes 01 02 03 ... 20 reversed: 20 1F 1E ... 01
static_assert(std::byteswap(make256(0x0102030405060708, 0x090A0B0C0D0E0F10, 0x1112131415161718, 0x191A1B1C1D1E1F20)) ==
              make256(0x201F1E1D1C1B1A19, 0x1817161514131211, 0x100F0E0D0C0B0A09, 0x0807060504030201));

// Double byteswap is identity
static_assert(std::byteswap(std::byteswap(make256(0xDEADBEEF, 0xCAFEBABE, 0x12345678, 0x9ABCDEF0))) ==
              make256(0xDEADBEEF, 0xCAFEBABE, 0x12345678, 0x9ABCDEF0));

// Signed byteswap compiles
static_assert(std::byteswap((__int256_t)0) == (__int256_t)0);

int main(int, char**) {
  // Runtime verification
  __uint256_t val      = make256(0x0102030405060708, 0x090A0B0C0D0E0F10, 0x1112131415161718, 0x191A1B1C1D1E1F20);
  __uint256_t swapped  = std::byteswap(val);
  __uint256_t expected = make256(0x201F1E1D1C1B1A19, 0x1817161514131211, 0x100F0E0D0C0B0A09, 0x0807060504030201);
  assert(swapped == expected);
  assert(std::byteswap(swapped) == val);

  return 0;
}
#endif
