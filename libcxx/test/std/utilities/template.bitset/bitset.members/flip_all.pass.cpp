//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// bitset<N>& flip(); // constexpr since C++23

#include <bitset>
#include <cassert>
#include <cstddef>
#include <vector>

#include "../bitset_test_cases.h"
#include "test_macros.h"

template <std::size_t N>
TEST_CONSTEXPR_CXX23 void test_flip_all() {
    std::vector<std::bitset<N> > const cases = get_test_cases<N>();
    for (std::size_t c = 0; c != cases.size(); ++c) {
        std::bitset<N> v1 = cases[c];
        std::bitset<N> v2 = v1;
        v2.flip();
        for (std::size_t i = 0; i < v1.size(); ++i)
            assert(v2[i] == ~v1[i]);
    }
}

TEST_CONSTEXPR_CXX23 bool test() {
  test_flip_all<0>();
  test_flip_all<1>();
  test_flip_all<31>();
  test_flip_all<32>();
  test_flip_all<33>();
  test_flip_all<63>();
  test_flip_all<64>();
  test_flip_all<65>();

  return true;
}

int main(int, char**) {
  test();
  test_flip_all<1000>(); // not in constexpr because of constexpr evaluation step limits
#if TEST_STD_VER > 20
  static_assert(test());
#endif

  return 0;
}
