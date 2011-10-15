//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// test constexpr bool test(size_t pos) const;

#include <bitset>
#include <cstdlib>
#include <cassert>

#pragma clang diagnostic ignored "-Wtautological-compare"

template <std::size_t N>
std::bitset<N>
make_bitset()
{
    std::bitset<N> v;
    for (std::size_t i = 0; i < N; ++i)
        v[i] = static_cast<bool>(std::rand() & 1);
    return v;
}

template <std::size_t N>
void test_test()
{
    const std::bitset<N> v1 = make_bitset<N>();
    try
    {
        bool b = v1.test(50);
        if (50 >= v1.size())
            assert(false);
        assert(b == v1[50]);
    }
    catch (std::out_of_range&)
    {
    }
}

int main()
{
    test_test<0>();
    test_test<1>();
    test_test<31>();
    test_test<32>();
    test_test<33>();
    test_test<63>();
    test_test<64>();
    test_test<65>();
    test_test<1000>();
}
