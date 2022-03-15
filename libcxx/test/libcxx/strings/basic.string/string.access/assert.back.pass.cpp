//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// Call back() on empty container.

// UNSUPPORTED: c++03, windows, libcxx-no-debug-mode
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=0

#include <string>

#include "check_assertion.h"
#include "min_allocator.h"

int main(int, char**) {
    {
        std::string s;
        TEST_LIBCPP_ASSERT_FAILURE(s.back(), "string::back(): string is empty");
    }

    {
        typedef std::basic_string<char, std::char_traits<char>, min_allocator<char> > S;
        S s;
        TEST_LIBCPP_ASSERT_FAILURE(s.back(), "string::back(): string is empty");
    }

    return 0;
}
