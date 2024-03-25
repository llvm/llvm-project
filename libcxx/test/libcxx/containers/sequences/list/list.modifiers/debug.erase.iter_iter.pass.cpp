//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <list>

// Call erase(const_iterator first, const_iterator last); with various invalid iterators

// REQUIRES: has-unix-headers
// UNSUPPORTED: !libcpp-has-legacy-debug-mode, c++03

#include <list>

#include "check_assertion.h"

int main(int, char**) {
    // First iterator from another container
    {
        int a1[] = {1, 2, 3};
        std::list<int> l1(a1, a1+3);
        std::list<int> l2(a1, a1+3);
        TEST_LIBCPP_ASSERT_FAILURE(l1.erase(l2.cbegin(), std::next(l1.cbegin())),
                                "list::erase(iterator, iterator) called with an iterator not referring to this list");
    }

    // Second iterator from another container
    {
        int a1[] = {1, 2, 3};
        std::list<int> l1(a1, a1+3);
        std::list<int> l2(a1, a1+3);
        TEST_LIBCPP_ASSERT_FAILURE(l1.erase(l1.cbegin(), std::next(l2.cbegin())),
                                "list::erase(iterator, iterator) called with an iterator not referring to this list");
    }

    // Both iterators from another container
    {
        int a1[] = {1, 2, 3};
        std::list<int> l1(a1, a1+3);
        std::list<int> l2(a1, a1+3);
        TEST_LIBCPP_ASSERT_FAILURE(l1.erase(l2.cbegin(), std::next(l2.cbegin())),
                                "list::erase(iterator, iterator) called with an iterator not referring to this list");
    }

    // With an invalid range
    {
        int a1[] = {1, 2, 3};
        std::list<int> l1(a1, a1+3);
        TEST_LIBCPP_ASSERT_FAILURE(l1.erase(std::next(l1.cbegin()), l1.cbegin()),
                                "Attempted to increment a non-incrementable list::const_iterator");
    }

    return 0;
}
