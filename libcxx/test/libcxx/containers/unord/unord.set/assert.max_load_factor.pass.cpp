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
// class unordered_set

// float max_load_factor() const;
// void max_load_factor(float mlf);

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03
// REQUIRES: libcpp-hardening-mode={{safe|debug}}
// XFAIL: availability-verbose_abort-missing

#include <unordered_set>

#include "check_assertion.h"

int main(int, char**) {
    typedef std::unordered_set<int> C;
    C c;
    TEST_LIBCPP_ASSERT_FAILURE(c.max_load_factor(-0.5f),
                               "unordered container::max_load_factor(lf) called with lf <= 0");

    return 0;
}
