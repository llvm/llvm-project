//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// XFAIL: libcpp-no-exceptions
// test bitset<N>& set(size_t pos, bool val = true);

#include <bitset>
#include <cassert>

template <std::size_t N>
void test_set_one()
{
    std::bitset<N> v;
    try
    {
        v.set(50);
        if (50 >= v.size())
            assert(false);
        assert(v[50]);
    }
    catch (std::out_of_range&)
    {
    }
    try
    {
        v.set(50, false);
        if (50 >= v.size())
            assert(false);
        assert(!v[50]);
    }
    catch (std::out_of_range&)
    {
    }
}

int main()
{
    test_set_one<0>();
    test_set_one<1>();
    test_set_one<31>();
    test_set_one<32>();
    test_set_one<33>();
    test_set_one<63>();
    test_set_one<64>();
    test_set_one<65>();
    test_set_one<1000>();
}
