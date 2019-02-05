//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <exception>

// class nested_exception;

// nested_exception(const nested_exception&) throw() = default;

#include <exception>
#include <cassert>

#include "test_macros.h"

class A
{
    int data_;
public:
    explicit A(int data) : data_(data) {}

    friend bool operator==(const A& x, const A& y) {return x.data_ == y.data_;}
};

int main(int, char**)
{
    {
        std::nested_exception e0;
        std::nested_exception e = e0;
        assert(e.nested_ptr() == nullptr);
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        try
        {
            throw A(2);
            assert(false);
        }
        catch (const A&)
        {
            std::nested_exception e0;
            std::nested_exception e = e0;
            assert(e.nested_ptr() != nullptr);
            try
            {
                rethrow_exception(e.nested_ptr());
                assert(false);
            }
            catch (const A& a)
            {
                assert(a == A(2));
            }
        }
    }
#endif

  return 0;
}
