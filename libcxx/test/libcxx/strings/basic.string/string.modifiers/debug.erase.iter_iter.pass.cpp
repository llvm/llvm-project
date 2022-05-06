//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// Call erase(const_iterator first, const_iterator last); with invalid iterators

// REQUIRES: has-unix-headers
// UNSUPPORTED: libcxx-no-debug-mode, c++03
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=1

#include <string>

#include "check_assertion.h"
#include "min_allocator.h"

int main(int, char**) {

    // With first iterator from another container
    {
        {
            std::string l1("123");
            std::string l2("123");
            TEST_LIBCPP_ASSERT_FAILURE(
                l1.erase(l2.cbegin(), l1.cbegin() + 1),
                "string::erase(iterator,  iterator) called with an iterator not referring to this string");
        }
        {
            typedef std::basic_string<char, std::char_traits<char>, min_allocator<char> > S;
            S l1("123");
            S l2("123");
            TEST_LIBCPP_ASSERT_FAILURE(l1.erase(l2.cbegin(), l1.cbegin() + 1),
                                        "string::erase(iterator,  iterator) called with an iterator not referring to this string");
        }
    }

    // With second iterator from another container
    {
        {
            std::string l1("123");
            std::string l2("123");
            TEST_LIBCPP_ASSERT_FAILURE(l1.erase(l1.cbegin(), l2.cbegin() + 1), "Attempted to compare incomparable iterators");
        }
        {
            typedef std::basic_string<char, std::char_traits<char>, min_allocator<char> > S;
            S l1("123");
            S l2("123");
            TEST_LIBCPP_ASSERT_FAILURE(l1.erase(l1.cbegin(), l2.cbegin() + 1), "Attempted to compare incomparable iterators");
        }
    }

    // With both iterators from another container
    {
        {
            std::string l1("123");
            std::string l2("123");
            TEST_LIBCPP_ASSERT_FAILURE(
                l1.erase(l2.cbegin(), l2.cbegin() + 1),
                "string::erase(iterator,  iterator) called with an iterator not referring to this string");
        }
        {
            typedef std::basic_string<char, std::char_traits<char>, min_allocator<char> > S;
            S l1("123");
            S l2("123");
            TEST_LIBCPP_ASSERT_FAILURE(
                l1.erase(l2.cbegin(), l2.cbegin() + 1),
                "string::erase(iterator,  iterator) called with an iterator not referring to this string");
        }
    }

    // With an invalid range
    {
        {
            std::string l1("123");
            TEST_LIBCPP_ASSERT_FAILURE(l1.erase(l1.cbegin() + 1, l1.cbegin()),
                                    "string::erase(first, last) called with invalid range");
        }
        {
            typedef std::basic_string<char, std::char_traits<char>, min_allocator<char> > S;
            S l1("123");
            TEST_LIBCPP_ASSERT_FAILURE(l1.erase(l1.cbegin() + 1, l1.cbegin()),
                                    "string::erase(first, last) called with invalid range");
        }
    }

    return 0;
}
