//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_map>

// Call erase(const_iterator first, const_iterator last); with invalid iterators

// REQUIRES: has-unix-headers
// UNSUPPORTED: !libcpp-has-debug-mode, c++03

#include <unordered_map>

#include "check_assertion.h"

int main(int, char**) {
    // First iterator from a different container
    {
        typedef std::pair<int, int> P;
        P a1[] = {P(1, 1), P(2, 2), P(3, 3)};
        std::unordered_map<int, int> l1(a1, a1+3);
        std::unordered_map<int, int> l2(a1, a1+3);
        TEST_LIBCPP_ASSERT_FAILURE(
            l1.erase(l2.cbegin(), std::next(l1.cbegin())),
            "unordered container::erase(iterator, iterator) called with an iterator not referring to this container");
    }

    // Second iterator from a different container
    {
        typedef std::pair<int, int> P;
        P a1[] = {P(1, 1), P(2, 2), P(3, 3)};
        std::unordered_map<int, int> l1(a1, a1+3);
        std::unordered_map<int, int> l2(a1, a1+3);
        TEST_LIBCPP_ASSERT_FAILURE(
            l1.erase(l1.cbegin(), std::next(l2.cbegin())),
            "unordered container::erase(iterator, iterator) called with an iterator not referring to this container");
    }

    // Both iterators from a different container
    {
        typedef std::pair<int, int> P;
        P a1[] = {P(1, 1), P(2, 2), P(3, 3)};
        std::unordered_map<int, int> l1(a1, a1+3);
        std::unordered_map<int, int> l2(a1, a1+3);
        TEST_LIBCPP_ASSERT_FAILURE(
            l1.erase(l2.cbegin(), std::next(l2.cbegin())),
            "unordered container::erase(iterator, iterator) called with an iterator not referring to this container");
    }

    // With iterators that don't form a valid range
    {
        typedef std::pair<int, int> P;
        P a1[] = {P(1, 1), P(2, 2), P(3, 3)};
        std::unordered_map<int, int> l1(a1, a1+3);
        TEST_LIBCPP_ASSERT_FAILURE(l1.erase(std::next(l1.cbegin()), l1.cbegin()),
                                "Attempted to increment a non-incrementable unordered container const_iterator");
    }

    return 0;
}
