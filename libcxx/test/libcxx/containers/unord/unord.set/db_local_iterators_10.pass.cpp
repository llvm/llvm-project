//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_set>

// Dereference non-dereferenceable iterator.

// UNSUPPORTED: libcxx-no-debug-mode
// UNSUPPORTED: c++03

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=1

#include <unordered_set>

#include "test_macros.h"
#include "debug_macros.h"
#include "min_allocator.h"

int main(int, char**) {
    typedef int T;
    typedef std::unordered_set<T, std::hash<T>, std::equal_to<T>, min_allocator<T>> C;
    C c(1);
    C::local_iterator i = c.end(0);
    TEST_LIBCPP_ASSERT_FAILURE(
        *i, "Attempted to dereference a non-dereferenceable unordered container const_local_iterator");

    return 0;
}
