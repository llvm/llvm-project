//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// XFAIL: libcpp-no-exceptions
// test bitset<N>& flip(size_t pos);

#include <bitset>
#include <cstdlib>
#include <cassert>

#if defined(__clang__)
#pragma clang diagnostic ignored "-Wtautological-compare"
#endif

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
void test_flip_one()
{
    std::bitset<N> v = make_bitset<N>();
    try
    {
        v.flip(50);
        bool b = v[50];
        if (50 >= v.size())
            assert(false);
        assert(v[50] == b);
        v.flip(50);
        assert(v[50] != b);
        v.flip(50);
        assert(v[50] == b);
    }
    catch (std::out_of_range&)
    {
    }
}

int main()
{
    test_flip_one<0>();
    test_flip_one<1>();
    test_flip_one<31>();
    test_flip_one<32>();
    test_flip_one<33>();
    test_flip_one<63>();
    test_flip_one<64>();
    test_flip_one<65>();
    test_flip_one<1000>();
}
