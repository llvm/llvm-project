//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_set>

// iterator insert(const_iterator p, const value_type& x);

// UNSUPPORTED: libcxx-no-debug-mode, c++03, windows
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=1

#include <unordered_set>

#include "check_assertion.h"

int main(int, char**) {
    typedef std::unordered_set<double> C;
    typedef C::value_type P;
    C c;
    C c2;
    C::const_iterator e = c2.end();
    P v(3.5);
    TEST_LIBCPP_ASSERT_FAILURE(
        c.insert(e, v),
        "unordered_set::insert(const_iterator, const value_type&) called with an iterator not referring to this unordered_set");

    return 0;
}
