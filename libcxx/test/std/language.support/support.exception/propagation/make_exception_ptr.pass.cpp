//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-no-exceptions
// <exception>

// template<class E> exception_ptr make_exception_ptr(E e);

#include <exception>
#include <cassert>

struct A
{
    static int constructed;
    int data_;

    A(int data = 0) : data_(data) {++constructed;}
    ~A() {--constructed;}
    A(const A& a) : data_(a.data_) {++constructed;}
};

int A::constructed = 0;

int main(int, char**)
{
    {
        std::exception_ptr p = std::make_exception_ptr(A(5));
        try
        {
            std::rethrow_exception(p);
            assert(false);
        }
        catch (const A& a)
        {
#ifndef _LIBCPP_ABI_MICROSOFT
            assert(A::constructed == 1);
#else
            // On Windows exception_ptr copies the exception
            assert(A::constructed == 2);
#endif
            assert(p != nullptr);
            p = nullptr;
            assert(p == nullptr);
            assert(a.data_ == 5);
            assert(A::constructed == 1);
        }
        assert(A::constructed == 0);
    }
    assert(A::constructed == 0);

  return 0;
}
