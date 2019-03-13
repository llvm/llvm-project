//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class X> class auto_ptr;

// explicit auto_ptr(X* p =0) throw();

#define _LIBCPP_DISABLE_DEPRECATION_WARNINGS

#include <memory>
#include <cassert>

#include "../A.h"

void
test()
{
    {
    A* p = new A(1);
    std::auto_ptr<A> ap = p;
    assert(ap.get() == p);
    assert(A::count == 1);
    }
    assert(A::count == 0);
    {
    std::auto_ptr<A> ap;
    assert(ap.get() == 0);
    }
}

int main(int, char**)
{
    test();

  return 0;
}
