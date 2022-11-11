//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// __unconstrained_reverse_iterator

// template <BidirectionalIterator Iter1, BidirectionalIterator Iter2>
//   requires HasEqualTo<Iter1, Iter2>
// bool operator==(const __unconstrained_reverse_iterator<Iter1>& x, const __unconstrained_reverse_iterator<Iter2>& y); // constexpr since C++17

#include <iterator>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"

template <class It>
TEST_CONSTEXPR_CXX17 void test(It l, It r, bool x) {
    const std::__unconstrained_reverse_iterator<It> r1(l);
    const std::__unconstrained_reverse_iterator<It> r2(r);
    assert((r1 == r2) == x);
}

TEST_CONSTEXPR_CXX17 bool tests() {
    const char* s = "1234567890";
    test(bidirectional_iterator<const char*>(s), bidirectional_iterator<const char*>(s), true);
    test(bidirectional_iterator<const char*>(s), bidirectional_iterator<const char*>(s+1), false);
    test(random_access_iterator<const char*>(s), random_access_iterator<const char*>(s), true);
    test(random_access_iterator<const char*>(s), random_access_iterator<const char*>(s+1), false);
    test(s, s, true);
    test(s, s+1, false);
    return true;
}

int main(int, char**) {
    tests();
#if TEST_STD_VER > 14
    static_assert(tests(), "");
#endif
    return 0;
}
