//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// Call back() on empty const container.

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03
// XFAIL: availability-verbose_abort-missing
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_ASSERTIONS=1

#include <string>

#include "check_assertion.h"
#include "min_allocator.h"

int main(int, char**) {
    {
        std::string const s;
        TEST_LIBCPP_ASSERT_FAILURE(s.back(), "string::back(): string is empty");
    }

    {
        typedef std::basic_string<char, std::char_traits<char>, min_allocator<char> > S;
        const S s;
        TEST_LIBCPP_ASSERT_FAILURE(s.back(), "string::back(): string is empty");
    }

    return 0;
}
