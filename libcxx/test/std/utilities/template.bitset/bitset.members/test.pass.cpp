//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// bool test(size_t pos) const; // constexpr since C++23

#include <bitset>
#include <cassert>
#include <cstddef>
#include <vector>

#include "../bitset_test_cases.h"
#include "test_macros.h"

template <std::size_t N>
TEST_CONSTEXPR_CXX23 void test_test() {
    std::vector<std::bitset<N> > const cases = get_test_cases<N>();
    for (std::size_t c = 0; c != cases.size(); ++c) {
        std::bitset<N> const v = cases[c];
        if (v.size() > 0) {
            std::size_t middle = v.size() / 2;
            bool b = v.test(middle);
            assert(b == v[middle]);
        }
    }
}

TEST_CONSTEXPR_CXX23 bool test() {
  test_test<0>();
  test_test<1>();
  test_test<31>();
  test_test<32>();
  test_test<33>();
  test_test<63>();
  test_test<64>();
  test_test<65>();

  return true;
}

int main(int, char**) {
  test();
  test_test<1000>(); // not in constexpr because of constexpr evaluation step limits
#if TEST_STD_VER > 20
  static_assert(test());
#endif

  return 0;
}
