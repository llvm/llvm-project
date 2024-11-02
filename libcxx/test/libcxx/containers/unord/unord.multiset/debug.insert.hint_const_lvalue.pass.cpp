//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_set>

// iterator insert(const_iterator p, const value_type& x);

// REQUIRES: has-unix-headers
// UNSUPPORTED: !libcpp-has-debug-mode, c++03

#include <unordered_set>

#include "check_assertion.h"

int main(int, char**) {
    typedef std::unordered_multiset<double> C;
    typedef C::value_type P;
    C c;
    C c2;
    C::const_iterator e = c2.end();
    P v(3.5);
    TEST_LIBCPP_ASSERT_FAILURE(
        c.insert(e, v),
        "unordered container::emplace_hint(const_iterator, args...) called with an iterator not referring to this unordered container");

    return 0;
}
