//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// reverse_iterator

// template <RandomAccessIterator Iterator>
// reverse_iterator<Iter> operator+(Iter::difference_type n, const reverse_iterator<Iter>& x); // constexpr in C++17

#include <iterator>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"

template <class It>
TEST_CONSTEXPR_CXX17 void test(It i, typename std::iterator_traits<It>::difference_type n, It x) {
    const std::reverse_iterator<It> r(i);
    std::reverse_iterator<It> rr = n + r;
    assert(rr.base() == x);
}

TEST_CONSTEXPR_CXX17 bool tests() {
    const char* s = "1234567890";
    test(random_access_iterator<const char*>(s+5), 5, random_access_iterator<const char*>(s));
#if TEST_STD_VER >= 20
    test(cpp20_random_access_iterator<const char*>(s + 5), 5, cpp20_random_access_iterator<const char*>(s));
#endif
    test(s+5, 5, s);
    return true;
}

int main(int, char**) {
    tests();
#if TEST_STD_VER > 14
    static_assert(tests(), "");
#endif
    return 0;
}
