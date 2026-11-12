//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// default ctor

#include <bitset>
#include <cassert>

#include "test_macros.h"

template <std::size_t N>
TEST_CONSTEXPR_CXX23 void test_default_ctor()
{
    {
        TEST_CONSTEXPR std::bitset<N> v1;
        assert(v1.size() == N);
        for (std::size_t i = 0; i < v1.size(); ++i)
            assert(v1[i] == false);
    }
#if TEST_STD_VER >= 11
    {
        constexpr std::bitset<N> v1;
        static_assert(v1.size() == N, "");
    }
#endif
}

TEST_CONSTEXPR_CXX23 bool test() {
  test_default_ctor<0>();
  test_default_ctor<1>();
  test_default_ctor<31>();
  test_default_ctor<32>();
  test_default_ctor<33>();
  test_default_ctor<63>();
  test_default_ctor<64>();
  test_default_ctor<65>();
  test_default_ctor<1000>();

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER > 20
  static_assert(test());
#endif

  return 0;
}
