//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-no-exceptions
// <exception>

// class nested_exception;

// void rethrow_nested [[noreturn]] () const;

#include <exception>
#include <cstdlib>
#include <cassert>

#include "test_macros.h"

class A
{
    int data_;
public:
    explicit A(int data) : data_(data) {}

    friend bool operator==(const A& x, const A& y) {return x.data_ == y.data_;}
};

void go_quietly()
{
    std::exit(0);
}

int main(int, char**)
{
    {
        try
        {
            throw A(2);
            assert(false);
        }
        catch (const A&)
        {
            const std::nested_exception e;
            assert(e.nested_ptr() != nullptr);
            try
            {
                e.rethrow_nested();
                assert(false);
            }
            catch (const A& a)
            {
                assert(a == A(2));
            }
        }
    }
    {
        try
        {
            std::set_terminate(go_quietly);
            const std::nested_exception e;
            e.rethrow_nested();
            assert(false);
        }
        catch (...)
        {
            assert(false);
        }
    }

  return 0;
}
