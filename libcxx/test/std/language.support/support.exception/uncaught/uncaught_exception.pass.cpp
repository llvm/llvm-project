//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-exceptions

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_CXX20_REMOVED_UNCAUGHT_EXCEPTION
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

// test uncaught_exception

#include <exception>
#include <cassert>

#include "test_macros.h"

struct A
{
    ~A()
    {
        assert(std::uncaught_exception());
    }
};

struct B
{
    B()
    {
        // http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#475
        assert(!std::uncaught_exception());
    }
};

int main(int, char**)
{
    try
    {
        A a;
        assert(!std::uncaught_exception());
        throw B();
    }
    catch (...)
    {
        assert(!std::uncaught_exception());
    }
    assert(!std::uncaught_exception());

  return 0;
}
