//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: 32-bit-pointer
// UNSUPPORTED: gcc

// Decimal formatting of __uint256_t requires division builtins from compiler-rt.
// ADDITIONAL_COMPILE_FLAGS: --rtlib=compiler-rt

// Test std::format support for __int256_t / __uint256_t

#include <cassert>
#include <format>
#include <string>

#include "test_macros.h"

#ifdef TEST_HAS_NO_INT256
int main(int, char**) { return 0; }
#else

int main(int, char**) {
  // Basic decimal formatting
  assert(std::format("{}", (__uint256_t)0) == "0");
  assert(std::format("{}", (__uint256_t)42) == "42");
  assert(std::format("{}", (__int256_t)-42) == "-42");
  assert(std::format("{}", (__int256_t)0) == "0");

  // Large values
  assert(std::format("{}", (__uint256_t)1 << 64) == "18446744073709551616");
  assert(std::format("{}", (__uint256_t)1 << 128) == "340282366920938463463374607431768211456");

  // Max value (2^256 - 1)
  assert(std::format("{}", (__uint256_t)-1) ==
         "115792089237316195423570985008687907853269984665640564039457584007913129639935");

  // Width and alignment
  assert(std::format("{:>5}", (__uint256_t)42) == "   42");
  assert(std::format("{:<5}", (__uint256_t)42) == "42   ");
  assert(std::format("{:^5}", (__uint256_t)42) == " 42  ");

  // Fill character
  assert(std::format("{:*>5}", (__uint256_t)42) == "***42");
  assert(std::format("{:0>5}", (__uint256_t)42) == "00042");

  // Sign
  assert(std::format("{:+}", (__int256_t)42) == "+42");
  assert(std::format("{:+}", (__int256_t)-42) == "-42");
  assert(std::format("{: }", (__int256_t)42) == " 42");

  // Hexadecimal
  assert(std::format("{:x}", (__uint256_t)255) == "ff");
  assert(std::format("{:X}", (__uint256_t)255) == "FF");
  assert(std::format("{:#x}", (__uint256_t)255) == "0xff");
  assert(std::format("{:#X}", (__uint256_t)255) == "0XFF");

  // Octal
  assert(std::format("{:o}", (__uint256_t)8) == "10");
  assert(std::format("{:#o}", (__uint256_t)8) == "010");

  // Binary
  assert(std::format("{:b}", (__uint256_t)10) == "1010");
  assert(std::format("{:#b}", (__uint256_t)10) == "0b1010");

  // Zero-padded with width
  assert(std::format("{:010}", (__uint256_t)42) == "0000000042");
  assert(std::format("{:010}", (__int256_t)-42) == "-000000042");

  // Comparison with __int128 formatting (should produce identical results
  // for values that fit in both types)
  __int128_t i128val = 123456789012345LL;
  __int256_t i256val = 123456789012345LL;
  assert(std::format("{}", i128val) == std::format("{}", i256val));
  assert(std::format("{:+020x}", i128val) == std::format("{:+020x}", i256val));

  // Full-width big-number tests (all 4 x 64-bit limbs populated).
  // Hex output directly corresponds to the hex digits of the input value.
  {
    __uint256_t big = ((__uint256_t)0xAAAABBBBCCCCDDDDULL << 192) | ((__uint256_t)0xEEEEFFFF11112222ULL << 128) |
                      ((__uint256_t)0x3333444455556666ULL << 64) | (__uint256_t)0x7777888899990000ULL;
    assert(std::format("{:x}", big) == "aaaabbbbccccddddeeeeffff1111222233334444555566667777888899990000");
    assert(std::format("{:X}", big) == "AAAABBBBCCCCDDDDEEEEFFFF1111222233334444555566667777888899990000");
    assert(std::format("{:#x}", big) == "0xaaaabbbbccccddddeeeeffff1111222233334444555566667777888899990000");
    // Width and alignment (64 hex digits, padded to 70)
    assert(std::format("{:>70x}", big) == "      aaaabbbbccccddddeeeeffff1111222233334444555566667777888899990000");
    assert(std::format("{:*<70x}", big) == "aaaabbbbccccddddeeeeffff1111222233334444555566667777888899990000******");
    // Zero-padded hex with prefix
    assert(std::format("{:#070x}", big) == "0x0000aaaabbbbccccddddeeeeffff1111222233334444555566667777888899990000");
  }

  // INT256_MIN: -(2^255).
  // Decimal verified: 2^256 = 11579...9936 (from UINT256_MAX + 1),
  // so 2^255 = 5789604461865809771178549250434395392663499233282028201972879200395656481
  //            9968
  {
    __uint256_t u_min  = (__uint256_t)1 << 255;
    __int256_t min_val = (__int256_t)u_min;
    assert(std::format("{}", min_val) ==
           "-57896044618658097711785492504343953926634992332820282019728792003956564819968");
  }

  // Large signed negative value in decimal (all limbs significant)
  {
    __int256_t neg = (__int256_t)-42;
    // Verify hex representation: -42 in hex is "-2a"
    assert(std::format("{:x}", neg) == "-2a");
    // Wide format of a negative value
    assert(std::format("{:+80}", neg) == std::string(77, ' ') + "-42");
  }

  return 0;
}
#endif
