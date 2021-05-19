//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// template<class InputIterator>
//   iterator insert(const_iterator p, InputIterator first, InputIterator last);

// UNSUPPORTED: libcxx-no-debug-mode

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=1

#include <string>

#include "test_macros.h"
#include "debug_macros.h"

int main(int, char**)
{
    std::string v;
    std::string v2;
    char a[] = "123";
    const int N = sizeof(a)/sizeof(a[0]);
    TEST_LIBCPP_ASSERT_FAILURE(v.insert(v2.cbegin() + 10, a, a + N),
                               "Attempted to add/subtract an iterator outside its valid range");

    return 0;
}
