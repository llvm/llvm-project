//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_map>

// iterator insert(const_iterator p, const value_type& x);

// UNSUPPORTED: libcxx-no-debug-mode

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=1

#include <unordered_map>

#include "test_macros.h"
#include "debug_macros.h"

int main(int, char**) {
    typedef std::unordered_multimap<double, int> C;
    typedef C::value_type P;
    C c;
    C c2;
    C::const_iterator e = c2.end();
    P v(3.5, 3);
    TEST_LIBCPP_ASSERT_FAILURE(
        c.insert(e, v),
        "unordered container::emplace_hint(const_iterator, args...) called with an iterator not referring to this unordered container");

    return 0;
}
