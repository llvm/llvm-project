//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>
//
// binder2nd

// UNSUPPORTED: clang-4.0
// UNSUPPORTED: c++98, c++03
// REQUIRES: verify-support

// MODULES_DEFINES: _LIBCPP_ENABLE_CXX17_REMOVED_BINDERS
#define _LIBCPP_ENABLE_CXX17_REMOVED_BINDERS

#include <functional>

#include "../test_func.h"
#include "test_macros.h"

int main(int, char**)
{
    typedef std::binder2nd<test_func> B2ND; // expected-error{{'binder2nd<test_func>' is deprecated}}

  return 0;
}
