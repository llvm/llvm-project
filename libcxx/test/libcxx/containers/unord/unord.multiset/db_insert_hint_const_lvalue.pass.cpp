//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_set>

// iterator insert(const_iterator p, const value_type& x);

// UNSUPPORTED: libcxx-no-debug-mode

#define _LIBCPP_DEBUG 1
#define _LIBCPP_ASSERT(x, m) ((x) ? (void)0 : std::exit(0))

#include <unordered_set>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        typedef std::unordered_multiset<double> C;
        typedef C::iterator R;
        typedef C::value_type P;
        C c;
        C c2;
        C::const_iterator e = c2.end();
        P v(3.5);
        R r = c.insert(e, v);
        assert(false);
    }

    return 0;
}
