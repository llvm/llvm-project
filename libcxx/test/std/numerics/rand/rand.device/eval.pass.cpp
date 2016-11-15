//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <random>

// class random_device;

// result_type operator()();

#include <random>
#include <cassert>

#include "test_macros.h"

int main()
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
        r();
        LIBCPP_ASSERT(false);
    }
    catch (const std::system_error&)
    {
    }
#endif
}
