//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// bitset<N>& operator<<=(size_t pos); // constexpr since C++23

#include <bitset>
#include <cassert>
#include <cstddef>
#include <vector>

#include "../bitset_test_cases.h"
#include "test_macros.h"

template <std::size_t N>
TEST_CONSTEXPR_CXX23 bool test_right_shift() {
    std::vector<std::bitset<N> > const cases = get_test_cases<N>();
    for (std::size_t c = 0; c != cases.size(); ++c) {
        for (std::size_t s = 0; s <= N+1; ++s) {
            std::bitset<N> v1 = cases[c];
            std::bitset<N> v2 = v1;
            v1 >>= s;
            for (std::size_t i = 0; i < v1.size(); ++i)
                if (i + s < v1.size())
                    assert(v1[i] == v2[i + s]);
                else
                    assert(v1[i] == 0);
        }
    }
    return true;
}

int main(int, char**) {
  test_right_shift<0>();
  test_right_shift<1>();
  test_right_shift<31>();
  test_right_shift<32>();
  test_right_shift<33>();
  test_right_shift<63>();
  test_right_shift<64>();
  test_right_shift<65>();
  test_right_shift<1000>(); // not in constexpr because of constexpr evaluation step limits
#if TEST_STD_VER > 20
  static_assert(test_right_shift<0>());
  static_assert(test_right_shift<1>());
  static_assert(test_right_shift<31>());
  static_assert(test_right_shift<32>());
  static_assert(test_right_shift<33>());
  static_assert(test_right_shift<63>());
  static_assert(test_right_shift<64>());
  static_assert(test_right_shift<65>());
#endif

  return 0;
}
