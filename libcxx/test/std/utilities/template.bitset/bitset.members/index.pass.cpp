//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// test bitset<N>::reference operator[](size_t pos);

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
void test_index_const()
{
    std::bitset<N> v1 = make_bitset<N>();
    if (N > 0)
    {
        assert(v1[N/2] == v1.test(N/2));
        typename std::bitset<N>::reference r = v1[N/2];
        assert(r == v1.test(N/2));
        typename std::bitset<N>::reference r2 = v1[N/2];
        r = r2;
        assert(r == v1.test(N/2));
        r = false;
        assert(r == false);
        assert(v1.test(N/2) == false);
        r = true;
        assert(r == true);
        assert(v1.test(N/2) == true);
        bool b = ~r;
        assert(r == true);
        assert(v1.test(N/2) == true);
        assert(b == false);
        r.flip();
        assert(r == false);
        assert(v1.test(N/2) == false);
    }
}

int main()
{
    test_index_const<0>();
    test_index_const<1>();
    test_index_const<31>();
    test_index_const<32>();
    test_index_const<33>();
    test_index_const<63>();
    test_index_const<64>();
    test_index_const<65>();
    test_index_const<1000>();
}
