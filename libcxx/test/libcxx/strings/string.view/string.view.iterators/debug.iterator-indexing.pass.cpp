//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Make sure that std::string_view's iterators check for OOB accesses when the debug mode is enabled.

// REQUIRES: has-unix-headers
// UNSUPPORTED: !libcpp-has-legacy-debug-mode

#include <string_view>

#include "check_assertion.h"

int main(int, char**) {
    // string_view::iterator
    {
        std::string_view const str("hello world");
        {
            auto it = str.end();
            TEST_LIBCPP_ASSERT_FAILURE(*it, "__bounded_iter::operator*: Attempt to dereference an out-of-range iterator");
        }
        {
            auto it = str.end();
            TEST_LIBCPP_ASSERT_FAILURE(it.operator->(), "__bounded_iter::operator->: Attempt to dereference an out-of-range iterator");
        }
        {
            auto it = str.begin();
            TEST_LIBCPP_ASSERT_FAILURE(it[99], "__bounded_iter::operator[]: Attempt to index an iterator out-of-range");
        }
    }

    // string_view::const_iterator
    {
        std::string_view const str("hello world");
        {
            auto it = str.cend();
            TEST_LIBCPP_ASSERT_FAILURE(*it, "__bounded_iter::operator*: Attempt to dereference an out-of-range iterator");
        }
        {
            auto it = str.cend();
            TEST_LIBCPP_ASSERT_FAILURE(it.operator->(), "__bounded_iter::operator->: Attempt to dereference an out-of-range iterator");
        }
        {
            auto it = str.cbegin();
            TEST_LIBCPP_ASSERT_FAILURE(it[99], "__bounded_iter::operator[]: Attempt to index an iterator out-of-range");
        }
    }

    // string_view::reverse_iterator
    {
        std::string_view const str("hello world");
        {
            auto it = str.rend();
            TEST_LIBCPP_ASSERT_FAILURE(*it, "__bounded_iter::operator*: Attempt to dereference an out-of-range iterator");
        }
        {
            auto it = str.rend();
            TEST_LIBCPP_ASSERT_FAILURE(it.operator->(), "__bounded_iter::operator->: Attempt to dereference an out-of-range iterator");
        }
        {
            auto it = str.rbegin();
            TEST_LIBCPP_ASSERT_FAILURE(it[99], "__bounded_iter::operator*: Attempt to dereference an out-of-range iterator");
        }
    }

    // string_view::const_reverse_iterator
    {
        std::string_view const str("hello world");
        {
            auto it = str.crend();
            TEST_LIBCPP_ASSERT_FAILURE(*it, "__bounded_iter::operator*: Attempt to dereference an out-of-range iterator");
        }
        {
            auto it = str.crend();
            TEST_LIBCPP_ASSERT_FAILURE(it.operator->(), "__bounded_iter::operator->: Attempt to dereference an out-of-range iterator");
        }
        {
            auto it = str.crbegin();
            TEST_LIBCPP_ASSERT_FAILURE(it[99], "__bounded_iter::operator*: Attempt to dereference an out-of-range iterator");
        }
    }

    return 0;
}
