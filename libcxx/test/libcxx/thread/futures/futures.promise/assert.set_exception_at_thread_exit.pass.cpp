//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03, no-threads
// REQUIRES: libcpp-hardening-mode={{extensive|debug}}
// XFAIL: availability-verbose_abort-missing

// <future>

// class promise<R>

// void set_exception_on_thread_exit(exception_ptr p);
// Test that a null exception_ptr is diagnosed.

#include <future>
#include <exception>

#include "check_assertion.h"

int main(int, char**) {
    {
        typedef int T;
        std::promise<T> p;
        TEST_LIBCPP_ASSERT_FAILURE(p.set_exception_at_thread_exit(std::exception_ptr()), "promise::set_exception_at_thread_exit: received nullptr");
    }

    {
        typedef int& T;
        std::promise<T> p;
        TEST_LIBCPP_ASSERT_FAILURE(p.set_exception_at_thread_exit(std::exception_ptr()), "promise::set_exception_at_thread_exit: received nullptr");
    }

    return 0;
}
