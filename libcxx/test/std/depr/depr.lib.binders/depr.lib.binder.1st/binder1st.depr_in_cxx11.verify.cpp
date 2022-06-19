//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>
//
// binder1st

// UNSUPPORTED: c++03
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_CXX17_REMOVED_BINDERS

#include <functional>

#include "../test_func.h"
#include "test_macros.h"

int main(int, char**)
{
    typedef std::binder1st<test_func> B1ST; // expected-warning {{'binder1st<test_func>' is deprecated}}

    return 0;
}
