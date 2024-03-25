//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <list>

// iterator insert(const_iterator position, size_type n, const value_type& x);

// REQUIRES: has-unix-headers
// UNSUPPORTED: !libcpp-has-legacy-debug-mode, c++03

#include <list>

#include "check_assertion.h"

int main(int, char**) {
    std::list<int> c1(100);
    std::list<int> c2;
    TEST_LIBCPP_ASSERT_FAILURE(c1.insert(c2.cbegin(), 5, 1),
                               "list::insert(iterator, n, x) called with an iterator not referring to this list");

    return 0;
}
