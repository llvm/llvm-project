//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <list>

// void pop_back();

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03
// XFAIL: availability-verbose_abort-missing
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_ASSERTIONS=1

#include <list>
#include <cassert>

#include "check_assertion.h"

int main(int, char**) {
    int a[] = {1, 2, 3};
    std::list<int> c(a, a+3);
    c.pop_back();
    assert(c == std::list<int>(a, a+2));
    c.pop_back();
    assert(c == std::list<int>(a, a+1));
    c.pop_back();
    assert(c.empty());
    TEST_LIBCPP_ASSERT_FAILURE(c.pop_back(), "list::pop_back() called on an empty list");

    return 0;
}
