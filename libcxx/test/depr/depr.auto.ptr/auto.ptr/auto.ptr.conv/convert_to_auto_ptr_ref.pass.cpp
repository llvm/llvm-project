//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class X> class auto_ptr;

// template<class Y> operator auto_ptr_ref<Y>() throw();

#include <memory>
#include <cassert>

#include "../AB.h"

void
test()
{
    B* p1 = new B(1);
    std::auto_ptr<B> ap1(p1);
    std::auto_ptr_ref<A> apr = ap1;
    delete p1;
}

int main()
{
    test();
}
