//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// iterator insert(const_iterator p, size_type n, charT c);

// UNSUPPORTED: libcxx-no-debug-mode, c++03, windows
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=1

#include <string>

#include "check_assertion.h"

int main(int, char**) {
    std::string s;
    std::string s2;
    TEST_LIBCPP_ASSERT_FAILURE(
        s.insert(s2.begin(), 1, 'a'),
        "string::insert(iterator, n, value) called with an iterator not referring to this string");

    return 0;
}
