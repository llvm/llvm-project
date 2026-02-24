//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// Test type traits support for __int256_t / __uint256_t

#include <cstddef>
#include <limits>
#include <type_traits>

#include "test_macros.h"

#ifdef TEST_HAS_NO_INT256
int main(int, char**) { return 0; }
#else

// is_integral
static_assert(std::is_integral<__int256_t>::value, "");
static_assert(std::is_integral<__uint256_t>::value, "");
static_assert(std::is_integral<const __int256_t>::value, "");
static_assert(std::is_integral<volatile __uint256_t>::value, "");

// is_arithmetic (derived from is_integral)
static_assert(std::is_arithmetic<__int256_t>::value, "");
static_assert(std::is_arithmetic<__uint256_t>::value, "");

// is_signed / is_unsigned
static_assert(std::is_signed<__int256_t>::value, "");
static_assert(!std::is_unsigned<__int256_t>::value, "");
static_assert(!std::is_signed<__uint256_t>::value, "");
static_assert(std::is_unsigned<__uint256_t>::value, "");

// is_fundamental
static_assert(std::is_fundamental<__int256_t>::value, "");
static_assert(std::is_fundamental<__uint256_t>::value, "");

// is_scalar
static_assert(std::is_scalar<__int256_t>::value, "");
static_assert(std::is_scalar<__uint256_t>::value, "");

// make_signed / make_unsigned
static_assert(std::is_same<std::make_signed<__uint256_t>::type, __int256_t>::value, "");
static_assert(std::is_same<std::make_signed<__int256_t>::type, __int256_t>::value, "");
static_assert(std::is_same<std::make_unsigned<__int256_t>::type, __uint256_t>::value, "");
static_assert(std::is_same<std::make_unsigned<__uint256_t>::type, __uint256_t>::value, "");

#  if TEST_STD_VER >= 14
static_assert(std::is_same<std::make_signed_t<__uint256_t>, __int256_t>::value, "");
static_assert(std::is_same<std::make_unsigned_t<__int256_t>, __uint256_t>::value, "");
#  endif

// numeric_limits
static_assert(std::numeric_limits<__int256_t>::is_specialized, "");
static_assert(std::numeric_limits<__uint256_t>::is_specialized, "");
static_assert(std::numeric_limits<__int256_t>::is_integer, "");
static_assert(std::numeric_limits<__uint256_t>::is_integer, "");
static_assert(std::numeric_limits<__int256_t>::is_signed, "");
static_assert(!std::numeric_limits<__uint256_t>::is_signed, "");
static_assert(std::numeric_limits<__int256_t>::digits == 255, ""); // 256 - 1 sign bit
static_assert(std::numeric_limits<__uint256_t>::digits == 256, "");
static_assert(std::numeric_limits<__int256_t>::is_exact, "");
static_assert(std::numeric_limits<__uint256_t>::radix == 2, "");

// sizeof
static_assert(sizeof(__int256_t) == 32, "");
static_assert(sizeof(__uint256_t) == 32, "");

// Comparison with __int128
static_assert(sizeof(__int256_t) == 2 * sizeof(__int128_t), "");
static_assert(std::numeric_limits<__uint256_t>::digits == 2 * std::numeric_limits<__uint128_t>::digits, "");

int main(int, char**) {
  // Runtime basic sanity
  __int256_t a  = 42;
  __uint256_t b = 100;
  __int256_t c  = a + (__int256_t)b;
  (void)c;

  // make_signed / make_unsigned runtime
  std::make_unsigned<__int256_t>::type u = 1;
  std::make_signed<__uint256_t>::type s  = -1;
  (void)u;
  (void)s;

  return 0;
}
#endif
