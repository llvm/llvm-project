//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03, no-threads
// XFAIL: availability-verbose_abort-missing
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_ASSERTIONS=1

// <future>

// class promise<R>

// void set_exception(exception_ptr p);
// Test that a null exception_ptr is diagnosed.

#include <future>
#include <exception>

#include "check_assertion.h"

int main(int, char**) {
    {
        typedef int T;
        std::promise<T> p;
        TEST_LIBCPP_ASSERT_FAILURE(p.set_exception(std::exception_ptr()), "promise::set_exception: received nullptr");
    }

    {
        typedef int& T;
        std::promise<T> p;
        TEST_LIBCPP_ASSERT_FAILURE(p.set_exception(std::exception_ptr()), "promise::set_exception: received nullptr");
    }

    return 0;
}
