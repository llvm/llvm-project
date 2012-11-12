//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// test bitset(string, pos, n, zero, one);

#include <bitset>
#include <cassert>

#pragma clang diagnostic ignored "-Wtautological-compare"

template <std::size_t N>
void test_string_ctor()
{
    {
    try
    {
        std::string str("xxx1010101010xxxx");
        std::bitset<N> v(str, str.size()+1, 10);
        assert(false);
    }
    catch (std::out_of_range&)
    {
    }
    }

    {
    try
    {
        std::string str("xxx1010101010xxxx");
        std::bitset<N> v(str, 2, 10);
        assert(false);
    }
    catch (std::invalid_argument&)
    {
    }
    }

    {
    std::string str("xxx1010101010xxxx");
    std::bitset<N> v(str, 3, 10);
    std::size_t M = std::min<std::size_t>(N, 10);
    for (std::size_t i = 0; i < M; ++i)
        assert(v[i] == (str[3 + M - 1 - i] == '1'));
    for (std::size_t i = 10; i < N; ++i)
        assert(v[i] == false);
    }

    {
    try
    {
        std::string str("xxxbababababaxxxx");
        std::bitset<N> v(str, 2, 10, 'a', 'b');
        assert(false);
    }
    catch (std::invalid_argument&)
    {
    }
    }

    {
    std::string str("xxxbababababaxxxx");
    std::bitset<N> v(str, 3, 10, 'a', 'b');
    std::size_t M = std::min<std::size_t>(N, 10);
    for (std::size_t i = 0; i < M; ++i)
        assert(v[i] == (str[3 + M - 1 - i] == 'b'));
    for (std::size_t i = 10; i < N; ++i)
        assert(v[i] == false);
    }
}

int main()
{
    test_string_ctor<0>();
    test_string_ctor<1>();
    test_string_ctor<31>();
    test_string_ctor<32>();
    test_string_ctor<33>();
    test_string_ctor<63>();
    test_string_ctor<64>();
    test_string_ctor<65>();
    test_string_ctor<1000>();
}
