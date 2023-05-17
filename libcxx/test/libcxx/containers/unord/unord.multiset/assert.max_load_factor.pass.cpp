//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_set>

// template <class Value, class Hash = hash<Value>, class Pred = equal_to<Value>,
//           class Alloc = allocator<Value>>
// class unordered_multiset

// float max_load_factor() const;
// void max_load_factor(float mlf);

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03
// XFAIL: availability-verbose_abort-missing
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_ASSERTIONS=1

#include <unordered_set>

#include "check_assertion.h"

int main(int, char**) {
    typedef std::unordered_multiset<int> C;
    C c;
    TEST_LIBCPP_ASSERT_FAILURE(c.max_load_factor(0), "unordered container::max_load_factor(lf) called with lf <= 0");

    return 0;
}
