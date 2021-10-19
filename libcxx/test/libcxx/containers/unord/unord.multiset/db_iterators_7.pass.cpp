//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_set>

// Increment iterator past end.

// UNSUPPORTED: libcxx-no-debug-mode

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=1

#include <unordered_set>
#include <cassert>

#include "test_macros.h"
#include "debug_macros.h"

int main(int, char**) {
    typedef int T;
    typedef std::unordered_multiset<T> C;
    C c;
    c.insert(42);
    C::iterator i = c.begin();
    assert(i != c.end());
    ++i;
    assert(i == c.end());
    TEST_LIBCPP_ASSERT_FAILURE(++i, "Attempted to increment a non-incrementable unordered container const_iterator");

    return 0;
}
