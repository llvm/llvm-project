//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test:

// bool operator==(const bitset<N>& rhs) const; // constexpr since C++23
// bool operator!=(const bitset<N>& rhs) const; // constexpr since C++23

#include <bitset>
#include <cassert>
#include <cstddef>
#include <vector>

#include "../bitset_test_cases.h"
#include "test_macros.h"

template <std::size_t N>
TEST_CONSTEXPR_CXX23 void test_equality() {
    std::vector<std::bitset<N> > const cases = get_test_cases<N>();
    for (std::size_t c = 0; c != cases.size(); ++c) {
        std::bitset<N> const v1 = cases[c];
        std::bitset<N> v2 = v1;
        assert(v1 == v2);
        if (v1.size() > 0) {
            v2[N/2].flip();
            assert(v1 != v2);
        }
    }
}

TEST_CONSTEXPR_CXX23 bool test() {
  test_equality<0>();
  test_equality<1>();
  test_equality<31>();
  test_equality<32>();
  test_equality<33>();
  test_equality<63>();
  test_equality<64>();
  test_equality<65>();

  return true;
}

int main(int, char**) {
  test();
  test_equality<1000>(); // not in constexpr because of constexpr evaluation step limits
#if TEST_STD_VER > 20
  static_assert(test());
#endif

  return 0;
}
