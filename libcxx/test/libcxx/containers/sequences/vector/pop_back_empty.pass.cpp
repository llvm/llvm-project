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

#include <vector>

#include "test_macros.h"
#include "debug_macros.h"

int main(int, char**) {
    std::vector<int> v;
    v.push_back(0);
    v.pop_back();
    TEST_LIBCPP_ASSERT_FAILURE(v.pop_back(), "vector::pop_back called on an empty vector");

    return 0;
}
