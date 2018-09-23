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
// bind1st

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
    // expected-error@* 1 {{'bind1st<test_func, int>' is deprecated}}
#endif
    std::bind1st(test_func(1), 5);
}
