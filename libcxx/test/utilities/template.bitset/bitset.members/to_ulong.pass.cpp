//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// test unsigned long to_ulong() const;

#include <bitset>
#include <algorithm>
#include <limits>
#include <climits>
#include <cassert>

template <std::size_t N>
void test_to_ulong()
{
    const std::size_t M = sizeof(unsigned long) * CHAR_BIT < N ? sizeof(unsigned long) * CHAR_BIT : N;
    const std::size_t X = M == 0 ? sizeof(unsigned long) * CHAR_BIT - 1 : sizeof(unsigned long) * CHAR_BIT - M;
    const std::size_t max = M == 0 ? 0 : std::size_t(std::numeric_limits<unsigned long>::max()) >> X;
    std::size_t tests[] = {0,
                           std::min<std::size_t>(1, max),
                           std::min<std::size_t>(2, max),
                           std::min<std::size_t>(3, max),
                           std::min(max, max-3),
                           std::min(max, max-2),
                           std::min(max, max-1),
                           max};
    for (std::size_t i = 0; i < sizeof(tests)/sizeof(tests[0]); ++i)
    {
        std::size_t j = tests[i];
        std::bitset<N> v(j);
        assert(j == v.to_ulong());
    }
}

int main()
{
    test_to_ulong<0>();
    test_to_ulong<1>();
    test_to_ulong<31>();
    test_to_ulong<32>();
    test_to_ulong<33>();
    test_to_ulong<63>();
    test_to_ulong<64>();
    test_to_ulong<65>();
    test_to_ulong<1000>();
}
