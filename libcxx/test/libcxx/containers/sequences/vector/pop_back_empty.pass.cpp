//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// pop_back() more than the number of elements in a vector

// UNSUPPORTED: libcxx-no-debug-mode

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=1
#define _LIBCPP_ASSERT(x, m) ((x) ? (void)0 : std::exit(0))

#include <cstdlib>
#include <vector>

#include "test_macros.h"


int main(int, char**) {
    std::vector<int> v;
    v.push_back(0);
    v.pop_back();
    v.pop_back();
    std::exit(1);

    return 0;
}
