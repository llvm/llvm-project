//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test size_t count() const;

#include <bitset>
#include <cassert>

template <std::size_t N>
void test_size()
{
    const std::bitset<N> v;
    assert(v.size() == N);
}

int main()
{
    test_size<0>();
    test_size<1>();
    test_size<31>();
    test_size<32>();
    test_size<33>();
    test_size<63>();
    test_size<64>();
    test_size<65>();
    test_size<1000>();
}
