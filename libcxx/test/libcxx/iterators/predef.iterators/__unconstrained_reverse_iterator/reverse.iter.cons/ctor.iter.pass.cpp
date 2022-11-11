//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// __unconstrained_reverse_iterator

// explicit __unconstrained_reverse_iterator(Iter x); // constexpr since C++17

#include <iterator>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"

template <class It>
TEST_CONSTEXPR_CXX17 void test(It i) {
    std::__unconstrained_reverse_iterator<It> r(i);
    assert(r.base() == i);
}

TEST_CONSTEXPR_CXX17 bool tests() {
    const char s[] = "123";
    test(bidirectional_iterator<const char*>(s));
    test(random_access_iterator<const char*>(s));
    test(s);
    return true;
}

int main(int, char**) {
    tests();
#if TEST_STD_VER > 14
    static_assert(tests(), "");
#endif
    return 0;
}
