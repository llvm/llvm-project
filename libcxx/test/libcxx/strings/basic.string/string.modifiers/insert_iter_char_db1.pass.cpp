//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// iterator insert(const_iterator p, charT c);

// UNSUPPORTED: libcxx-no-debug-mode

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=1

#include <string>

#include "test_macros.h"
#include "debug_macros.h"

int main(int, char**)
{
    typedef std::string S;
    S s;
    S s2;
    TEST_LIBCPP_ASSERT_FAILURE(s.insert(s2.begin(), '1'), "Attempted to subtract incompatible iterators");

    return 0;
}
