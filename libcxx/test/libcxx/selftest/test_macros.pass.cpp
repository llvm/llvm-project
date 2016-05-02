//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Test the "test_macros.h" header.
#include "test_macros.h"

#ifndef TEST_STD_VER
#error TEST_STD_VER must be defined
#endif

#ifndef TEST_NOEXCEPT
#error TEST_NOEXCEPT must be defined
#endif

#ifndef LIBCPP_ASSERT
#error LIBCPP_ASSERT must be defined
#endif

#ifndef LIBCPP_STATIC_ASSERT
#error LIBCPP_STATIC_ASSERT must be defined
#endif

void test_noexcept() TEST_NOEXCEPT
{
}

int main()
{
    test_noexcept();
}
