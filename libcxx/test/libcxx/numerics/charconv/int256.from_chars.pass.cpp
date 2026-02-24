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

// Test std::from_chars support for __int256_t / __uint256_t.
//
// from_chars works generically for all integral types via SFINAE on
// is_integral<_Tp>::value. The implementation uses __itoa::__traits<_Tp>
// for the base-10 fast path, and __itoa::__mul_overflowed (via
// __builtin_mul_overflow) for other bases. Both support __uint256_t.

#include <charconv>
#include <cstring>
#include <limits>

#include "test_macros.h"

#ifdef TEST_HAS_NO_INT256
int main(int, char**) { return 0; }
#else

// Helper: round-trip through to_chars then from_chars, verify value is preserved.
template <typename T>
bool round_trip(T value) {
  char buf[80];
  auto [to_ptr, to_ec] = std::to_chars(buf, buf + sizeof(buf), value);
  if (to_ec != std::errc{})
    return false;

  T parsed{};
  auto [from_ptr, from_ec] = std::from_chars(buf, to_ptr, parsed);
  if (from_ec != std::errc{})
    return false;
  if (from_ptr != to_ptr)
    return false;
  return parsed == value;
}

// Helper: round-trip with explicit base.
template <typename T>
bool round_trip_base(T value, int base) {
  char buf[260]; // base-2 of 256-bit = 256 chars + sign
  auto [to_ptr, to_ec] = std::to_chars(buf, buf + sizeof(buf), value, base);
  if (to_ec != std::errc{})
    return false;

  T parsed{};
  auto [from_ptr, from_ec] = std::from_chars(buf, to_ptr, parsed, base);
  if (from_ec != std::errc{})
    return false;
  if (from_ptr != to_ptr)
    return false;
  return parsed == value;
}

int main(int, char**) {
  // ====================================================================
  // Basic from_chars (base 10, default)
  // ====================================================================

  // --- Parse small unsigned values ---
  {
    __uint256_t val;
    const char* str = "42";
    auto [ptr, ec]  = std::from_chars(str, str + 2, val);
    if (ec != std::errc{} || val != 42 || ptr != str + 2)
      return 1;
  }

  // --- Parse zero ---
  {
    __uint256_t val;
    const char* str = "0";
    auto [ptr, ec]  = std::from_chars(str, str + 1, val);
    if (ec != std::errc{} || val != 0)
      return 2;
  }

  // --- Parse negative signed value ---
  {
    __int256_t val;
    const char* str = "-1";
    auto [ptr, ec]  = std::from_chars(str, str + 2, val);
    if (ec != std::errc{} || val != -1)
      return 3;
  }

  // --- Parse value > 64-bit ---
  {
    __uint256_t val;
    const char* str = "18446744073709551616"; // 2^64
    auto [ptr, ec]  = std::from_chars(str, str + std::strlen(str), val);
    if (ec != std::errc{} || val != ((__uint256_t)1 << 64))
      return 4;
  }

  // --- Parse value > 128-bit ---
  {
    __uint256_t val;
    // 2^128 = 340282366920938463463374607431768211456
    const char* str = "340282366920938463463374607431768211456";
    auto [ptr, ec]  = std::from_chars(str, str + std::strlen(str), val);
    if (ec != std::errc{} || val != ((__uint256_t)1 << 128))
      return 5;
  }

  // --- Invalid input ---
  {
    __uint256_t val = 999;
    const char* str = "abc";
    auto [ptr, ec]  = std::from_chars(str, str + 3, val);
    if (ec != std::errc::invalid_argument)
      return 6;
    // val should be unchanged on error
  }

  // --- Leading zeros ---
  {
    __uint256_t val;
    const char* str = "00042";
    auto [ptr, ec]  = std::from_chars(str, str + 5, val);
    if (ec != std::errc{} || val != 42)
      return 7;
  }

  // ====================================================================
  // Round-trip: to_chars -> from_chars for various values
  // ====================================================================

  // Unsigned values
  if (!round_trip<__uint256_t>(0))
    return 10;
  if (!round_trip<__uint256_t>(1))
    return 11;
  if (!round_trip<__uint256_t>(42))
    return 12;
  if (!round_trip<__uint256_t>((__uint256_t)1 << 64))
    return 13;
  if (!round_trip<__uint256_t>((__uint256_t)1 << 128))
    return 14;
  if (!round_trip<__uint256_t>((__uint256_t)1 << 200))
    return 15;
  if (!round_trip<__uint256_t>(~(__uint256_t)0)) // max
    return 16;

  // Signed values
  if (!round_trip<__int256_t>(0))
    return 20;
  if (!round_trip<__int256_t>(1))
    return 21;
  if (!round_trip<__int256_t>(-1))
    return 22;
  if (!round_trip<__int256_t>((__int256_t)1 << 200))
    return 23;
  if (!round_trip<__int256_t>(std::numeric_limits<__int256_t>::max()))
    return 24;
  if (!round_trip<__int256_t>(std::numeric_limits<__int256_t>::min()))
    return 25;

  // ====================================================================
  // Non-decimal bases: hex, octal, binary
  // ====================================================================

  // --- Hex (base 16) ---
  {
    __uint256_t val;
    const char* str = "ff";
    auto [ptr, ec]  = std::from_chars(str, str + 2, val, 16);
    if (ec != std::errc{} || val != 255)
      return 30;
  }

  // --- Hex round-trip ---
  if (!round_trip_base<__uint256_t>((__uint256_t)1 << 128, 16))
    return 31;
  if (!round_trip_base<__uint256_t>(~(__uint256_t)0, 16))
    return 32;

  // --- Octal (base 8) ---
  {
    __uint256_t val;
    const char* str = "777";
    auto [ptr, ec]  = std::from_chars(str, str + 3, val, 8);
    if (ec != std::errc{} || val != 0777)
      return 33;
  }

  // --- Binary (base 2) ---
  {
    __uint256_t val;
    const char* str = "1010";
    auto [ptr, ec]  = std::from_chars(str, str + 4, val, 2);
    if (ec != std::errc{} || val != 10)
      return 34;
  }

  // --- Base 36 ---
  if (!round_trip_base<__uint256_t>((__uint256_t)1 << 100, 36))
    return 35;

  // ====================================================================
  // Overflow detection
  // ====================================================================

  // --- Unsigned overflow ---
  {
    __uint256_t val;
    // max uint256 + 1 in decimal: append a digit to max
    // Use a string that's definitely too large
    const char* str = "115792089237316195423570985008687907853"
                      "269984665640564039457584007913129639936"; // 2^256
    auto [ptr, ec]  = std::from_chars(str, str + std::strlen(str), val);
    if (ec != std::errc::result_out_of_range)
      return 40;
  }

  // --- Signed overflow (positive) ---
  {
    __int256_t val;
    // max int256 + 1 = 2^255
    const char* str = "57896044618658097711785492504343953926"
                      "634992332820282019728792003956564819968";
    auto [ptr, ec]  = std::from_chars(str, str + std::strlen(str), val);
    if (ec != std::errc::result_out_of_range)
      return 41;
  }

  return 0;
}
#endif
