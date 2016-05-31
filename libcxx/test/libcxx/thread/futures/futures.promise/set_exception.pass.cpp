//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//

// UNSUPPORTED: libcpp-no-exceptions
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: c++98, c++03

// <future>

// class promise<R>

// void set_exception(exception_ptr p);
// Test that a null exception_ptr is diagnosed.

#define _LIBCPP_ASSERT(x, m) ((x) ? ((void)0) : throw 42)

#define _LIBCPP_DEBUG 0
#include <future>
#include <exception>
#include <cstdlib>
#include <cassert>


int main()
{
    {
        typedef int T;
        std::promise<T> p;
        try {
            p.set_exception(std::exception_ptr());
            assert(false);
        } catch (int const& value) {
            assert(value == 42);
        }
    }
    {
        typedef int& T;
        std::promise<T> p;
        try {
            p.set_exception(std::exception_ptr());
            assert(false);
        } catch (int const& value) {
            assert(value == 42);
        }
    }
}
