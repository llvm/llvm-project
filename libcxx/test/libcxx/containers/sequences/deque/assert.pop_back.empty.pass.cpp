//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <deque>

// pop_back() more than the number of elements in a deque

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03
// UNSUPPORTED: libcpp-hardening-mode=none
// XFAIL: availability-verbose_abort-missing

#include <deque>

#include "check_assertion.h"

int main(int, char**) {
    std::deque<int> q;
    q.push_back(0);
    q.pop_back();
    TEST_LIBCPP_ASSERT_FAILURE(q.pop_back(), "deque::pop_back called on an empty deque");

    return 0;
}
