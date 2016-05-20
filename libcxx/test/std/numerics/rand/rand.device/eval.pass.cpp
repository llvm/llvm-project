//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// XFAIL: libcpp-no-exceptions
// <random>

// class random_device;

// result_type operator()();

#include <random>
#include <cassert>

int main()
{
    {
        std::random_device r;
        std::random_device::result_type e = r();
    }

    try
    {
        std::random_device r("/dev/null");
        r();
        assert(false);
    }
    catch (const std::system_error&)
    {
    }
}
