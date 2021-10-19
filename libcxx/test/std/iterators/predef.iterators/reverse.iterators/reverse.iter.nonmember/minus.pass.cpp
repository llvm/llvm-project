//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// reverse_iterator

// template <RandomAccessIterator Iter1, RandomAccessIterator Iter2>
//   requires HasMinus<Iter2, Iter1>
// auto operator-(const reverse_iterator<Iter1>& x, const reverse_iterator<Iter2>& y) // constexpr in C++17
//  -> decltype(y.base() - x.base());

#include <iterator>
#include <cstddef>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"

template <class It1, class It2>
TEST_CONSTEXPR_CXX17 void test(It1 l, It2 r, std::ptrdiff_t x) {
    const std::reverse_iterator<It1> r1(l);
    const std::reverse_iterator<It2> r2(r);
    assert((r1 - r2) == x);
}

TEST_CONSTEXPR_CXX17 bool tests() {
    char s[3] = {0};
    test(random_access_iterator<const char*>(s), random_access_iterator<char*>(s), 0);
    test(random_access_iterator<char*>(s), random_access_iterator<const char*>(s+1), 1);
    test(random_access_iterator<const char*>(s+1), random_access_iterator<char*>(s), -1);
    test(s, s, 0);
    test(s, s+1, 1);
    test(s+1, s, -1);
    return true;
}

int main(int, char**) {
    tests();
#if TEST_STD_VER > 14
    static_assert(tests(), "");
#endif
    return 0;
}
