//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// XFAIL: libcpp-no-exceptions
// test bitset<N>& reset(size_t pos);

#include <bitset>
#include <cassert>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wtautological-compare"
#endif

template <std::size_t N>
void test_reset_one()
{
    std::bitset<N> v;
    try
    {
        v.set();
        v.reset(50);
        if (50 >= v.size())
            assert(false);
        for (unsigned i = 0; i < v.size(); ++i)
            if (i == 50)
                assert(!v[i]);
            else
                assert(v[i]);
    }
    catch (std::out_of_range&)
    {
    }
}

int main()
{
    test_reset_one<0>();
    test_reset_one<1>();
    test_reset_one<31>();
    test_reset_one<32>();
    test_reset_one<33>();
    test_reset_one<63>();
    test_reset_one<64>();
    test_reset_one<65>();
    test_reset_one<1000>();
}
