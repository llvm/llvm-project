//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class X> class auto_ptr;

// void reset(X* p=0) throw();

// REQUIRES: c++98 || c++03 || c++11 || c++14

#define _LIBCPP_DISABLE_DEPRECATION_WARNINGS

#include <memory>
#include <cassert>

#include "test_macros.h"
#include "../A.h"

void
test()
{
    {
    A* p = new A(1);
    std::auto_ptr<A> ap(p);
    ap.reset();
    assert(ap.get() == 0);
    assert(A::count == 0);
    }
    assert(A::count == 0);
    {
    A* p = new A(1);
    std::auto_ptr<A> ap(p);
    ap.reset(p);
    assert(ap.get() == p);
    assert(A::count == 1);
    }
    assert(A::count == 0);
    {
    A* p = new A(1);
    std::auto_ptr<A> ap(p);
    A* p2 = new A(2);
    ap.reset(p2);
    assert(ap.get() == p2);
    assert(A::count == 1);
    }
    assert(A::count == 0);
}

int main(int, char**)
{
    test();

  return 0;
}
