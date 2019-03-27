//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// See bugs.llvm.org/PR20183
//
// XFAIL: with_system_cxx_lib=macosx10.11
// XFAIL: with_system_cxx_lib=macosx10.10
// XFAIL: with_system_cxx_lib=macosx10.9
// XFAIL: with_system_cxx_lib=macosx10.8
// XFAIL: with_system_cxx_lib=macosx10.7

// <random>

// class random_device;

// result_type operator()();

#include <random>
#include <cassert>
#include <system_error>

#include "test_macros.h"

int main(int, char**)
{
    {
        std::random_device r;
        std::random_device::result_type e = r();
        ((void)e); // Prevent unused warning
    }

#ifndef TEST_HAS_NO_EXCEPTIONS
    try
    {
        std::random_device r("/dev/null");
        (void)r();
        LIBCPP_ASSERT(false);
    }
    catch (const std::system_error&)
    {
    }
#endif

  return 0;
}
