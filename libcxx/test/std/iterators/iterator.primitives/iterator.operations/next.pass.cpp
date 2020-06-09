//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// template <InputIterator Iter>
//   Iter next(Iter x, Iter::difference_type n = 1); // constexpr in C++17

// LWG #2353 relaxed the requirement on next from ForwardIterator to InputIterator

#include <iterator>
#include <cassert>
#include <type_traits>

#include "test_macros.h"
#include "test_iterators.h"

template <class It>
TEST_CONSTEXPR_CXX17 void
check_next_n(It it, typename std::iterator_traits<It>::difference_type n, It result)
{
    static_assert(std::is_same<decltype(std::next(it, n)), It>::value, "");
    assert(std::next(it, n) == result);

    It (*next_ptr)(It, typename std::iterator_traits<It>::difference_type) = std::next;
    assert(next_ptr(it, n) == result);
}

template <class It>
TEST_CONSTEXPR_CXX17 void
check_next_1(It it, It result)
{
    static_assert(std::is_same<decltype(std::next(it)), It>::value, "");
    assert(std::next(it) == result);
}

TEST_CONSTEXPR_CXX17 bool tests()
{
    const char* s = "1234567890";
    check_next_n(input_iterator<const char*>(s),             10, input_iterator<const char*>(s+10));
    check_next_n(forward_iterator<const char*>(s),           10, forward_iterator<const char*>(s+10));
    check_next_n(bidirectional_iterator<const char*>(s),     10, bidirectional_iterator<const char*>(s+10));
    check_next_n(bidirectional_iterator<const char*>(s+10), -10, bidirectional_iterator<const char*>(s));
    check_next_n(random_access_iterator<const char*>(s),     10, random_access_iterator<const char*>(s+10));
    check_next_n(random_access_iterator<const char*>(s+10), -10, random_access_iterator<const char*>(s));
    check_next_n(s, 10, s+10);

    check_next_1(input_iterator<const char*>(s), input_iterator<const char*>(s+1));
    check_next_1(forward_iterator<const char*>(s), forward_iterator<const char*>(s+1));
    check_next_1(bidirectional_iterator<const char*>(s), bidirectional_iterator<const char*>(s+1));
    check_next_1(random_access_iterator<const char*>(s), random_access_iterator<const char*>(s+1));
    check_next_1(s, s+1);

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
