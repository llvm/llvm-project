//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_set>

// Increment local_iterator past end.

// UNSUPPORTED: libcxx-no-debug-mode
// UNSUPPORTED: c++03

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=1

#include <unordered_set>
#include <cassert>

#include "test_macros.h"
#include "debug_macros.h"
#include "min_allocator.h"

int main(int, char**) {
    typedef int T;
    typedef std::unordered_multiset<T, std::hash<T>, std::equal_to<T>, min_allocator<T>> C;
    C c({42});
    C::size_type b = c.bucket(42);
    C::local_iterator i = c.begin(b);
    assert(i != c.end(b));
    ++i;
    assert(i == c.end(b));
    TEST_LIBCPP_ASSERT_FAILURE(++i,
                               "Attempted to increment a non-incrementable unordered container const_local_iterator");

    return 0;
}
