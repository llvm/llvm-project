//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test bool any() const;

#include <bitset>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

template <std::size_t N>
void test_any()
{
    std::bitset<N> v;
    v.reset();
    assert(v.any() == false);
    v.set();
    assert(v.any() == (N != 0));
    const bool greater_than_1 = std::integral_constant<bool, (N > 1)>::value; // avoid compiler warnings
    if (greater_than_1)
    {
        v[N/2] = false;
        assert(v.any() == true);
        v.reset();
        v[N/2] = true;
        assert(v.any() == true);
    }
}

int main(int, char**)
{
    test_any<0>();
    test_any<1>();
    test_any<31>();
    test_any<32>();
    test_any<33>();
    test_any<63>();
    test_any<64>();
    test_any<65>();
    test_any<1000>();

  return 0;
}
