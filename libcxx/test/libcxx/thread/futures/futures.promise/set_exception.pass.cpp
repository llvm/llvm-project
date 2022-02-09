//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: windows
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: c++03
// UNSUPPORTED: libcxx-no-debug-mode

// ADDITIONAL_COMPILE_FLAGS: -Wno-macro-redefined -D_LIBCPP_DEBUG=0

// <future>

// class promise<R>

// void set_exception(exception_ptr p);
// Test that a null exception_ptr is diagnosed.

#include <future>
#include <exception>
#include <cstdlib>
#include <cassert>

#include "test_macros.h"
#include "debug_mode_helper.h"

int main(int, char**)
{
    {
        typedef int T;
        std::promise<T> p;

        EXPECT_DEATH( p.set_exception(std::exception_ptr()) );
    }
    {
        typedef int& T;
        std::promise<T> p;

        EXPECT_DEATH( p.set_exception(std::exception_ptr()) );
    }

  return 0;
}
