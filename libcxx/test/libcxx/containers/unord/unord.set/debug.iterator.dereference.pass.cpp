//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_set>

// Dereference non-dereferenceable iterator.

// REQUIRES: has-unix-headers
// UNSUPPORTED: !libcpp-has-legacy-debug-mode, c++03

#include <unordered_set>

#include "check_assertion.h"
#include "min_allocator.h"

int main(int, char**) {
    {
        typedef int T;
        typedef std::unordered_set<T> C;
        C c(1);
        C::iterator i = c.end();
        TEST_LIBCPP_ASSERT_FAILURE(*i, "Attempted to dereference a non-dereferenceable unordered container const_iterator");
    }

    {
        typedef int T;
        typedef std::unordered_set<T, std::hash<T>, std::equal_to<T>, min_allocator<T>> C;
        C c(1);
        C::iterator i = c.end();
        TEST_LIBCPP_ASSERT_FAILURE(*i, "Attempted to dereference a non-dereferenceable unordered container const_iterator");
    }

    return 0;
}
