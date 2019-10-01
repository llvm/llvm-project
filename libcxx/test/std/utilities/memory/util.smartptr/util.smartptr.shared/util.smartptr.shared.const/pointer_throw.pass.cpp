//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-no-exceptions
// UNSUPPORTED: sanitizer-new-delete

// <memory>

// template<class Y> explicit shared_ptr(Y* p);


#include <memory>
#include <new>
#include <cstdlib>
#include <cassert>

#include "count_new.hpp"

#include "test_macros.h"

struct A
{
    static int count;

    A() {++count;}
    A(const A&) {++count;}
    ~A() {--count;}
};

int A::count = 0;


int main(int, char**)
{
    A* ptr = new A;
    assert(A::count == 1);
    globalMemCounter.throw_after = 0;
    try
    {
        std::shared_ptr<A> p(ptr);
        assert(false);
    }
    catch (std::bad_alloc&)
    {
        assert(A::count == 0);
    }
    assert(globalMemCounter.checkOutstandingNewEq(0));

  return 0;
}
