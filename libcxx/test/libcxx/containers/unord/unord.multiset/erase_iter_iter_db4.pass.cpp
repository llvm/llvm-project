//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_set>

// Call erase(const_iterator first, const_iterator last); with a bad range

// UNSUPPORTED: libcxx-no-debug-mode

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=1

#include <unordered_set>

#include "test_macros.h"
#include "debug_macros.h"

int main(int, char**) {
    int a1[] = {1, 2, 3};
    std::unordered_multiset<int> l1(a1, a1+3);
    TEST_LIBCPP_ASSERT_FAILURE(l1.erase(std::next(l1.cbegin()), l1.cbegin()),
                               "Attempted to increment a non-incrementable unordered container const_iterator");

    return 0;
}
