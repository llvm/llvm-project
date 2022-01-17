//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-has-no-random-device

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

    // When using the `/dev/urandom` implementation, make sure that we throw
    // an exception when we hit EOF while reading the custom-provided file.
#if !defined(TEST_HAS_NO_EXCEPTIONS) && defined(_LIBCPP_USING_DEV_RANDOM)
    {
        std::random_device r("/dev/null");
        try {
            (void)r();
            LIBCPP_ASSERT(false);
        } catch (const std::system_error&) {
        }
    }
#endif

  return 0;
}
