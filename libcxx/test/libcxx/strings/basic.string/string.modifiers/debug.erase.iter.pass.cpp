//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// Call erase(const_iterator position) with an iterator from another container

// REQUIRES: has-unix-headers
// UNSUPPORTED: !libcpp-has-debug-mode, c++03

#include <string>

#include "check_assertion.h"
#include "min_allocator.h"

int main(int, char**) {
    {
        std::string l1("123");
        std::string l2("123");
        std::string::const_iterator i = l2.begin();
        TEST_LIBCPP_ASSERT_FAILURE(l1.erase(i), "string::erase(iterator) called with an iterator not referring to this string");
    }

    {
        typedef std::basic_string<char, std::char_traits<char>, min_allocator<char> > S;
        S l1("123");
        S l2("123");
        S::const_iterator i = l2.begin();
        TEST_LIBCPP_ASSERT_FAILURE(l1.erase(i), "string::erase(iterator) called with an iterator not referring to this string");
    }

    return 0;
}
