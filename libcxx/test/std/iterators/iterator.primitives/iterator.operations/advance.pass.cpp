//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

//   All of these became constexpr in C++17
//
// template <InputIterator Iter>
//   constexpr void advance(Iter& i, Iter::difference_type n);
//
// template <BidirectionalIterator Iter>
//   constexpr void advance(Iter& i, Iter::difference_type n);
//
// template <RandomAccessIterator Iter>
//   constexpr void advance(Iter& i, Iter::difference_type n);

#include <iterator>
#include <cassert>
#include <type_traits>

#include "test_macros.h"
#include "test_iterators.h"

template <class It>
TEST_CONSTEXPR_CXX17
void check_advance(It it, typename std::iterator_traits<It>::difference_type n, It result)
{
    static_assert(std::is_same<decltype(std::advance(it, n)), void>::value, "");
    std::advance(it, n);
    assert(it == result);
}

TEST_CONSTEXPR_CXX17 bool tests()
{
    const char* s = "1234567890";
    check_advance(input_iterator<const char*>(s), 10, input_iterator<const char*>(s+10));
    check_advance(forward_iterator<const char*>(s), 10, forward_iterator<const char*>(s+10));
    check_advance(bidirectional_iterator<const char*>(s+5), 5, bidirectional_iterator<const char*>(s+10));
    check_advance(bidirectional_iterator<const char*>(s+5), -5, bidirectional_iterator<const char*>(s));
    check_advance(random_access_iterator<const char*>(s+5), 5, random_access_iterator<const char*>(s+10));
    check_advance(random_access_iterator<const char*>(s+5), -5, random_access_iterator<const char*>(s));
    check_advance(s+5, 5, s+10);
    check_advance(s+5, -5, s);

    return true;
}

int main(int, char**)
{
    tests();
#if TEST_STD_VER >= 17
    static_assert(tests(), "");
#endif
    return 0;
}
