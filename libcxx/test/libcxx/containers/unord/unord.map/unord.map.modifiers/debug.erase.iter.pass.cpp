//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_map>

// Call erase(const_iterator position) with invalid iterators

// REQUIRES: has-unix-headers
// UNSUPPORTED: libcxx-no-debug-mode, c++03
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=1

#include <unordered_map>

#include "check_assertion.h"

int main(int, char**) {
    // With end()
    {
        typedef std::pair<int, int> P;
        P a1[] = {P(1, 1), P(2, 2), P(3, 3)};
        std::unordered_map<int, int> l1(a1, a1+3);
        std::unordered_map<int, int>::const_iterator i = l1.end();
        TEST_LIBCPP_ASSERT_FAILURE(l1.erase(i),
                                "unordered container erase(iterator) called with a non-dereferenceable iterator");
    }

    // With iterator from another container
    {
        typedef std::pair<int, int> P;
        P a1[] = {P(1, 1), P(2, 2), P(3, 3)};
        std::unordered_map<int, int> l1(a1, a1+3);
        std::unordered_map<int, int> l2(a1, a1+3);
        std::unordered_map<int, int>::const_iterator i = l2.begin();
        TEST_LIBCPP_ASSERT_FAILURE(
            l1.erase(i), "unordered container erase(iterator) called with an iterator not referring to this container");
    }

    return 0;
}
