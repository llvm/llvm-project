//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// __unconstrained_reverse_iterator

// __unconstrained_reverse_iterator operator--(int); // constexpr in C++17

#include <iterator>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"

template <class It>
TEST_CONSTEXPR_CXX17 void test(It i, It x) {
    std::__unconstrained_reverse_iterator<It> r(i);
    std::__unconstrained_reverse_iterator<It> rr = r--;
    assert(r.base() == x);
    assert(rr.base() == i);
}

TEST_CONSTEXPR_CXX17 bool tests() {
    const char* s = "123";
    test(bidirectional_iterator<const char*>(s+1), bidirectional_iterator<const char*>(s+2));
    test(random_access_iterator<const char*>(s+1), random_access_iterator<const char*>(s+2));
    test(s+1, s+2);
    return true;
}

int main(int, char**) {
    tests();
#if TEST_STD_VER > 14
    static_assert(tests(), "");
#endif
    return 0;
}
