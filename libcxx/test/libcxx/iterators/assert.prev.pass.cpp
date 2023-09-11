//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03
// UNSUPPORTED: !libcpp-has-debug-mode && !libcpp-has-assertions
// XFAIL: availability-verbose_abort-missing

// <list>

// Call prev(forward_iterator, -1)

#include <iterator>

#include "check_assertion.h"
#include "test_iterators.h"

int main(int, char**) {
    int a[] = {1, 2, 3};

    bidirectional_iterator<int *> bidi(a+1);
    std::prev(bidi, -1);  // should work fine
    std::prev(bidi,  0);  // should work fine
    std::prev(bidi,  1);  // should work fine

    forward_iterator<int *> it(a+1);
    std::prev(it, -1); // should work fine
    std::prev(it,  0); // should work fine
    TEST_LIBCPP_ASSERT_FAILURE(std::prev(it, 1), "Attempt to prev(it, n) with a positive n on a non-bidirectional iterator");

    return 0;
}
