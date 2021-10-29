//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-monotonic-clock

// TODO(ldionne): This test fails on Ubuntu Focal on our CI nodes (and only there), in 32 bit mode.
// UNSUPPORTED: linux && 32bits-on-64bits

// <chrono>

// steady_clock

// static time_point now();

#include <chrono>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    typedef std::chrono::steady_clock C;
    C::time_point t1 = C::now();
    C::time_point t2 = C::now();
    assert(t2 >= t1);
    // make sure t2 didn't wrap around
    assert(t2 > std::chrono::time_point<C>());

  return 0;
}
