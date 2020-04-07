//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: windows
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: c++98, c++03

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=0

// This test requires debug mode, which the library on macOS doesn't have.
// UNSUPPORTED: with_system_cxx_lib=macosx

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
