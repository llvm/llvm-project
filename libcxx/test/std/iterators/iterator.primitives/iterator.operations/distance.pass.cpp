//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// template <InputIterator Iter>
//   Iter::difference_type
//   distance(Iter first, Iter last); // constexpr in C++17
//
// template <RandomAccessIterator Iter>
//   Iter::difference_type
//   distance(Iter first, Iter last); // constexpr in C++17

#include <iterator>
#include <cassert>
#include <type_traits>

#include "test_macros.h"
#include "test_iterators.h"

template <class It>
TEST_CONSTEXPR_CXX17
void check_distance(It first, It last, typename std::iterator_traits<It>::difference_type dist)
{
    typedef typename std::iterator_traits<It>::difference_type Difference;
    static_assert(std::is_same<decltype(std::distance(first, last)), Difference>::value, "");
    assert(std::distance(first, last) == dist);
}

TEST_CONSTEXPR_CXX17 bool tests()
{
    const char* s = "1234567890";
    check_distance(input_iterator<const char*>(s), input_iterator<const char*>(s+10), 10);
    check_distance(forward_iterator<const char*>(s), forward_iterator<const char*>(s+10), 10);
    check_distance(bidirectional_iterator<const char*>(s), bidirectional_iterator<const char*>(s+10), 10);
    check_distance(random_access_iterator<const char*>(s), random_access_iterator<const char*>(s+10), 10);
    check_distance(s, s+10, 10);
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
