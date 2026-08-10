//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// bitset<N>& reset(size_t pos); // constexpr since C++23

#include <bitset>
#include <cassert>
#include <cstddef>
#include <vector>

#include "../bitset_test_cases.h"
#include "test_macros.h"

TEST_MSVC_DIAGNOSTIC_IGNORED(6294) // Ill-defined for-loop:  initial condition does not satisfy test.  Loop body not executed.

template <std::size_t N>
TEST_CONSTEXPR_CXX23 void test_reset_one() {
    std::vector<std::bitset<N> > const cases = get_test_cases<N>();
    for (std::size_t c = 0; c != cases.size(); ++c) {
        for (std::size_t i = 0; i != N; ++i) {
            std::bitset<N> v = cases[c];
            v.reset(i);
            assert(v[i] == false);
        }
    }
}

TEST_CONSTEXPR_CXX23 bool test() {
  test_reset_one<0>();
  test_reset_one<1>();
  test_reset_one<31>();
  test_reset_one<32>();
  test_reset_one<33>();
  test_reset_one<63>();
  test_reset_one<64>();
  test_reset_one<65>();

  return true;
}

int main(int, char**) {
  test();
  test_reset_one<1000>(); // not in constexpr because of constexpr evaluation step limits
#if TEST_STD_VER > 20
  static_assert(test());
#endif

  return 0;
}
