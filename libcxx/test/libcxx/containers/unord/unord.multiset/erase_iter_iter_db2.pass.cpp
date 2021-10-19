//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_set>

// Call erase(const_iterator first, const_iterator last); with second iterator from another container

// UNSUPPORTED: libcxx-no-debug-mode

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=1

#include <unordered_set>

#include "test_macros.h"
#include "debug_macros.h"

int main(int, char**) {
    int a1[] = {1, 2, 3};
    std::unordered_multiset<int> l1(a1, a1+3);
    std::unordered_multiset<int> l2(a1, a1+3);
    TEST_LIBCPP_ASSERT_FAILURE(
        l1.erase(l1.cbegin(), std::next(l2.cbegin())),
        "unordered container::erase(iterator, iterator) called with an iterator not referring to this container");

    return 0;
}
