//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: c++03

// <random>

// template<class _IntType = int>
// class uniform_int_distribution

// template<class _URNG> result_type operator()(_URNG& g);


#include <cassert>
#include <cstdint>
#include <limits>
#include <random>

#include "test_macros.h"

int main(int, char**) {

#ifndef TEST_HAS_NO_INT128

  // Test that values outside of the 64-bit range can be produced.
  {
    std::minstd_rand0 e;
    std::uniform_int_distribution<__int128_t> d;
    assert(d.min() == 0 && d.max() == std::numeric_limits<__int128_t>::max());
    bool all_in_64bit_range = true;
    for (int i = 0; i < 100; ++i) {
      __int128_t n = d(e);
      all_in_64bit_range = all_in_64bit_range && (n <= UINT64_MAX);
    }
    assert(!all_in_64bit_range);
  }

  // Same test as above with min/max set and outside the 64-bit range.
  {
    __int128_t a = ((__int128_t)INT64_MIN) * 10;
    __int128_t b = ((__int128_t)INT64_MAX) * 10;
    std::minstd_rand0 e;
    std::uniform_int_distribution<__int128_t> d(a, b);
    assert(d.min() == a && d.max() == b);
    bool all_in_64bit_range = true;
    for (int i = 0; i < 100; ++i) {
      __int128_t n = d(e);
      assert(a <= n && n <= b);
      all_in_64bit_range = all_in_64bit_range && (INT64_MIN <= n) && (n <= (INT64_MAX));
    }
    assert(!all_in_64bit_range);
  }

  // Same test as above with __uint128_t.
  {
    __uint128_t a = UINT64_MAX / 3;
    __uint128_t b = ((__uint128_t)UINT64_MAX) * 10;
    std::minstd_rand0 e;
    std::uniform_int_distribution<__uint128_t> d(a, b);
    assert(d.min() == a && d.max() == b);
    bool all_in_64bit_range = true;
    for (int i = 0; i < 100; ++i) {
      __uint128_t n = d(e);
      assert(a <= n && n <= b);
      all_in_64bit_range = all_in_64bit_range && (n <= (UINT64_MAX));
    }
    assert(!all_in_64bit_range);
  }

  // Regression test for PR#51520:
  {
    std::minstd_rand0 e;
    std::uniform_int_distribution<__int128_t> d(INT64_MIN, INT64_MAX);
    assert(d.min() == INT64_MIN && d.max() == INT64_MAX);
    for (int i = 0; i < 100; ++i) {
      __int128_t n = d(e);
      assert((INT64_MIN <= n) && (n <= INT64_MAX));
    }
  }

#endif // TEST_HAS_NO_INT128

  return 0;
}
