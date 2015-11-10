//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// XFAIL: libcpp-no-exceptions
// test uncaught_exceptions

#include <exception>
#include <cassert>

struct A
{
    ~A()
    {
        assert(std::uncaught_exceptions() > 0);
    }
};

struct B
{
    B()
    {
        // http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#475
        assert(std::uncaught_exceptions() == 0);
    }
};

int main()
{
    try
    {
        A a;
        assert(std::uncaught_exceptions() == 0);
        throw B();
    }
    catch (...)
    {
        assert(std::uncaught_exception() == 0);
    }
    assert(std::uncaught_exceptions() == 0);
}
