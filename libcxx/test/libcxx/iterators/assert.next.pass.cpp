//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03
// REQUIRES: libcpp-hardening-mode={{safe|debug}}
// XFAIL: availability-verbose_abort-missing

// <list>

// Call next(non-bidi iterator, -1)

#include <iterator>

#include "check_assertion.h"
#include "test_iterators.h"

int main(int, char**) {
    int a[] = {1, 2, 3};
    forward_iterator<int *> it(a+1);
    std::next(it, 1);  // should work fine
    std::next(it, 0);  // should work fine
    TEST_LIBCPP_ASSERT_FAILURE(std::next(it, -1), "Attempt to next(it, n) with negative n on a non-bidirectional iterator");

    return 0;
}
