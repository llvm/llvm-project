//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-filesystem-library

// Filesystem is supported on Apple platforms starting with macosx10.15.
// UNSUPPORTED: with_system_cxx_lib=macosx10.14
// UNSUPPORTED: with_system_cxx_lib=macosx10.13
// UNSUPPORTED: with_system_cxx_lib=macosx10.12
// UNSUPPORTED: with_system_cxx_lib=macosx10.11
// UNSUPPORTED: with_system_cxx_lib=macosx10.10
// UNSUPPORTED: with_system_cxx_lib=macosx10.9

// XFAIL: LIBCXX-WINDOWS-FIXME

// <chrono>

// file_clock

// static time_point now() noexcept;

#include <chrono>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    typedef std::chrono::file_clock C;
    ASSERT_NOEXCEPT(C::now());

    C::time_point t1 = C::now();
    assert(t1.time_since_epoch().count() != 0);
    assert(C::time_point::min() < t1);
    assert(C::time_point::max() > t1);

  return 0;
}
