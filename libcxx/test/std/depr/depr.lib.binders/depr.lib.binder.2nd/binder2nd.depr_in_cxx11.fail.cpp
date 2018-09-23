//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <functional>
//
// binder2nd

// REQUIRES: verify-support

// MODULES_DEFINES: _LIBCPP_ENABLE_DEPRECATION_WARNINGS
// MODULES_DEFINES: _LIBCPP_ENABLE_CXX17_REMOVED_BINDERS
#define _LIBCPP_ENABLE_DEPRECATION_WARNINGS
#define _LIBCPP_ENABLE_CXX17_REMOVED_BINDERS

#include <functional>

#include "../test_func.h"
#include "test_macros.h"

int main()
{
#if TEST_STD_VER < 11
    // expected-no-diagnostics
#else
    // expected-error@* 1 {{'binder2nd<test_func>' is deprecated}}
#endif
    typedef std::binder2nd<test_func> B2ND;
}
