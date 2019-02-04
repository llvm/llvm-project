//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class X> class auto_ptr;

// template<class Y> operator auto_ptr_ref<Y>() throw();

// REQUIRES: c++98 || c++03 || c++11 || c++14

#include <memory>
#include <cassert>

#include "../AB.h"

void
test()
{
    B* p1 = new B(1);
    std::auto_ptr<B> ap1(p1);
    std::auto_ptr_ref<A> apr = ap1;
    std::auto_ptr<A> ap2(apr);
    assert(ap1.get() == nullptr);
    assert(ap2.get() == p1);
    ap2.release();
    delete p1;
}

int main()
{
    test();
}
