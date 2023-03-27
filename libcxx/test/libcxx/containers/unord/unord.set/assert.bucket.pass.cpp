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

// size_type bucket(const key_type& __k) const;

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03
// XFAIL: availability-verbose_abort-missing
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_ASSERTIONS=1

#include <unordered_set>

#include "check_assertion.h"

int main(int, char**) {
    typedef std::unordered_set<int> C;
    C c;
    TEST_LIBCPP_ASSERT_FAILURE(c.bucket(3), "unordered container::bucket(key) called when bucket_count() == 0");

    return 0;
}
