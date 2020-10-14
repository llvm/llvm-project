//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_map>

// Increment iterator past end.

// UNSUPPORTED: libcxx-no-debug-mode

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=1
#define _LIBCPP_ASSERT(x, m) ((x) ? (void)0 : std::exit(0))

#include <unordered_map>
#include <cassert>
#include <string>
#include <utility>

#include "test_macros.h"

int main(int, char**) {
    typedef std::unordered_multimap<int, std::string> C;
    C c;
    c.insert(std::make_pair(1, "one"));
    C::iterator i = c.begin();
    ++i;
    assert(i == c.end());
    ++i;
    assert(false);

    return 0;
}
