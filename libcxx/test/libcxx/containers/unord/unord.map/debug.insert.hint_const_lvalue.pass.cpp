//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_map>

// iterator insert(const_iterator p, const value_type& x);

// UNSUPPORTED: libcxx-no-debug-mode, c++03, windows
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=1

#include <unordered_map>

#include "check_assertion.h"
#include "test_macros.h"

int main(int, char**) {
    typedef std::unordered_map<double, int> C;
    typedef C::value_type P;
    C c;
    C c2;
    C::const_iterator e = c2.end();
    P v(3.5, 3);
#if TEST_STD_VER < 11
    TEST_LIBCPP_ASSERT_FAILURE(
        c.insert(e, v),
        "unordered_map::insert(const_iterator, const value_type&) called with an iterator not referring to this unordered_map");
#else
    TEST_LIBCPP_ASSERT_FAILURE(
        c.insert(e, v),
        "unordered_map::insert(const_iterator, value_type&&) called with an iterator not referring to this unordered_map");
#endif

    return 0;
}
