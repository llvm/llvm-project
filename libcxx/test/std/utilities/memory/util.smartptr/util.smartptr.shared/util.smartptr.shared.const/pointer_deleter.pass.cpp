//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// shared_ptr

// template<class Y, class D> shared_ptr(Y* p, D d);

#include <memory>
#include <cassert>
#include "deleter_types.h"

struct A
{
    static int count;

    A() {++count;}
    A(const A&) {++count;}
    ~A() {--count;}
};

int A::count = 0;

int main()
{
    {
    A* ptr = new A;
    std::shared_ptr<A> p(ptr, test_deleter<A>(3));
    assert(A::count == 1);
    assert(p.use_count() == 1);
    assert(p.get() == ptr);
    test_deleter<A>* d = std::get_deleter<test_deleter<A> >(p);
    assert(test_deleter<A>::count == 1);
    assert(test_deleter<A>::dealloc_count == 0);
    assert(d);
    assert(d->state() == 3);
    }
    assert(A::count == 0);
    assert(test_deleter<A>::count == 0);
    assert(test_deleter<A>::dealloc_count == 1);
}
