//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <unordered_map>

// template <class P,
//           class = typename enable_if<is_convertible<P, value_type>::value>::type>
//     iterator insert(const_iterator p, P&& x);

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
    TEST_LIBCPP_ASSERT_FAILURE(
        c.insert(e, P(3.5, 3)),
        "unordered container::emplace_hint(const_iterator, args...) called with an iterator not referring to this unordered container");

    return 0;
}
