//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// unsigned long long to_ullong() const; // constexpr since C++23

#include <bitset>
#include <algorithm>
#include <type_traits>
#include <limits>
#include <climits>
#include <cassert>
#include <stdexcept>

#include "test_macros.h"

template <std::size_t N>
TEST_CONSTEXPR_CXX23 void test_to_ullong() {
  const std::size_t M  = sizeof(unsigned long long) * CHAR_BIT < N ? sizeof(unsigned long long) * CHAR_BIT : N;
  const bool is_M_zero = std::integral_constant < bool, M == 0 > ::value; // avoid compiler warnings
  const std::size_t X =
      is_M_zero ? sizeof(unsigned long long) * CHAR_BIT - 1 : sizeof(unsigned long long) * CHAR_BIT - M;
  const unsigned long long max = is_M_zero ? 0 : (unsigned long long)(-1) >> X;
  unsigned long long tests[]   = {
      0,
      std::min<unsigned long long>(1, max),
      std::min<unsigned long long>(2, max),
      std::min<unsigned long long>(3, max),
      std::min(max, max - 3),
      std::min(max, max - 2),
      std::min(max, max - 1),
      max};
  for (unsigned long long j : tests) {
    std::bitset<N> v(j);
    assert(j == v.to_ullong());
  }
  { // test values bigger than can fit into the bitset
    const unsigned long long val = 0x55AAAAFFFFAAAA55ULL;
    const bool canFit            = N < sizeof(unsigned long long) * CHAR_BIT;
    const unsigned long long mask =
        canFit ? (1ULL << (canFit ? N : 0)) - 1 : (unsigned long long)(-1); // avoid compiler warnings
    std::bitset<N> v(val);
    assert(v.to_ullong() == (val & mask)); // we shouldn't return bit patterns from outside the limits of the bitset.
  }
}

TEST_CONSTEXPR_CXX23 bool test() {
  test_to_ullong<0>();
  test_to_ullong<1>();
  test_to_ullong<31>();
  test_to_ullong<32>();
  test_to_ullong<33>();
  test_to_ullong<63>();
  test_to_ullong<64>();
  test_to_ullong<65>();
  test_to_ullong<1000>();

#ifndef TEST_HAS_NO_EXCEPTIONS
  if (!TEST_IS_CONSTANT_EVALUATED) {
    // bitset has true bits beyond the size of unsigned long long
    std::bitset<std::numeric_limits<unsigned long long>::digits + 1> q(0);
    q.flip();
    try {
      (void)q.to_ullong(); // throws
      assert(false);
    } catch (const std::overflow_error&) {
      // expected
    } catch (...) {
      assert(false);
    }
  }
#endif // TEST_HAS_NO_EXCEPTIONS

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER > 20
  static_assert(test());
#endif

  return 0;
}
