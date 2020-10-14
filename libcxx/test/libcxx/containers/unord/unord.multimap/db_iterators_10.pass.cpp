//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_map>

// Dereference non-dereferenceable iterator.

// UNSUPPORTED: libcxx-no-debug-mode
// UNSUPPORTED: c++03

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=1
#define _LIBCPP_ASSERT(x, m) ((x) ? (void)0 : std::exit(0))

#include <cassert>
#include <functional>
#include <string>
#include <unordered_map>
#include <utility>

#include "test_macros.h"
#include "min_allocator.h"

int main(int, char**) {
    typedef std::unordered_multimap<int, std::string, std::hash<int>, std::equal_to<int>,
                        min_allocator<std::pair<const int, std::string>>> C;
    C c;
    c.insert(std::make_pair(1, "one"));
    C::iterator i = c.end();
    C::value_type j = *i;
    assert(false);

    return 0;
}
