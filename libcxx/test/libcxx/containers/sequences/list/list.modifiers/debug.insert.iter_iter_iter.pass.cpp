//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <list>

// template <InputIterator Iter>
//   iterator insert(const_iterator position, Iter first, Iter last);

// REQUIRES: has-unix-headers
// UNSUPPORTED: !libcpp-has-legacy-debug-mode, c++03

#include <list>

#include "check_assertion.h"

int main(int, char**) {
    std::list<int> v(100);
    std::list<int> v2(100);
    int a[] = {1, 2, 3, 4, 5};
    TEST_LIBCPP_ASSERT_FAILURE(v.insert(v2.cbegin(), a, a + 5),
                                "list::insert(iterator, range) called with an iterator not referring to this list");

    return 0;
}
