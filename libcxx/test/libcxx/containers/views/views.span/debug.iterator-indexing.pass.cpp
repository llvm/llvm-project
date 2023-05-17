//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// Make sure that std::span's iterators check for OOB accesses when the debug mode is enabled.

// REQUIRES: has-unix-headers
// UNSUPPORTED: !libcpp-has-debug-mode

#include <span>

#include "check_assertion.h"

struct Foo {
    int x;
};

int main(int, char**) {
    // span<T>::iterator
    {
        Foo array[] = {{0}, {1}, {2}};
        std::span<Foo> const span(array, 3);
        {
            auto it = span.end();
            TEST_LIBCPP_ASSERT_FAILURE(*it, "__bounded_iter::operator*: Attempt to dereference an out-of-range iterator");
        }
        {
            auto it = span.end();
            TEST_LIBCPP_ASSERT_FAILURE(it->x, "__bounded_iter::operator->: Attempt to dereference an out-of-range iterator");
        }
        {
            auto it = span.begin();
            TEST_LIBCPP_ASSERT_FAILURE(it[3], "__bounded_iter::operator[]: Attempt to index an iterator out-of-range");
        }
    }

    // span<T, N>::iterator
    {
        Foo array[] = {{0}, {1}, {2}};
        std::span<Foo, 3> const span(array, 3);
        {
            auto it = span.end();
            TEST_LIBCPP_ASSERT_FAILURE(*it, "__bounded_iter::operator*: Attempt to dereference an out-of-range iterator");
        }
        {
            auto it = span.end();
            TEST_LIBCPP_ASSERT_FAILURE(it->x, "__bounded_iter::operator->: Attempt to dereference an out-of-range iterator");
        }
        {
            auto it = span.begin();
            TEST_LIBCPP_ASSERT_FAILURE(it[3], "__bounded_iter::operator[]: Attempt to index an iterator out-of-range");
        }
    }

    // span<T>::reverse_iterator
    {
        Foo array[] = {{0}, {1}, {2}};
        std::span<Foo> const span(array, 3);
        {
            auto it = span.rend();
            TEST_LIBCPP_ASSERT_FAILURE(*it, "__bounded_iter::operator*: Attempt to dereference an out-of-range iterator");
        }
        {
            auto it = span.rend();
            TEST_LIBCPP_ASSERT_FAILURE(it->x, "__bounded_iter::operator->: Attempt to dereference an out-of-range iterator");
        }
        {
            auto it = span.rbegin();
            TEST_LIBCPP_ASSERT_FAILURE(it[3], "__bounded_iter::operator*: Attempt to dereference an out-of-range iterator");
        }
    }

    // span<T, N>::reverse_iterator
    {
        Foo array[] = {{0}, {1}, {2}};
        std::span<Foo, 3> const span(array, 3);
        {
            auto it = span.rend();
            TEST_LIBCPP_ASSERT_FAILURE(*it, "__bounded_iter::operator*: Attempt to dereference an out-of-range iterator");
        }
        {
            auto it = span.rend();
            TEST_LIBCPP_ASSERT_FAILURE(it->x, "__bounded_iter::operator->: Attempt to dereference an out-of-range iterator");
        }
        {
            auto it = span.rbegin();
            TEST_LIBCPP_ASSERT_FAILURE(it[3], "__bounded_iter::operator*: Attempt to dereference an out-of-range iterator");
        }
    }

    return 0;
}
