//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// template<class Y> explicit shared_ptr(auto_ptr<Y>&& r);

// REQUIRES: c++03 || c++11 || c++14
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

#include <memory>
#include <new>
#include <cstdlib>
#include <cassert>

#include "test_macros.h"
#include "count_new.h"

struct B
{
    static int count;

    B() {++count;}
    B(const B&) {++count;}
    virtual ~B() {--count;}
};

int B::count = 0;

struct A
    : public B
{
    static int count;

    A() {++count;}
    A(const A& other) : B(other) {++count;}
    ~A() {--count;}
};

int A::count = 0;

int main(int, char**)
{
    globalMemCounter.reset();
    {
        std::auto_ptr<A> ptr(new A);
        A* raw_ptr = ptr.get();
        std::shared_ptr<B> p(std::move(ptr));
        assert(A::count == 1);
        assert(B::count == 1);
        assert(p.use_count() == 1);
        assert(p.get() == raw_ptr);
        assert(ptr.get() == 0);
    }
    assert(A::count == 0);
    assert(globalMemCounter.checkOutstandingNewEq(0));

    globalMemCounter.reset();
    {
        std::auto_ptr<A const> ptr(new A);
        A const* raw_ptr = ptr.get();
        std::shared_ptr<B const> p(std::move(ptr));
        assert(A::count == 1);
        assert(B::count == 1);
        assert(p.use_count() == 1);
        assert(p.get() == raw_ptr);
        assert(ptr.get() == 0);
    }
    assert(A::count == 0);
    assert(globalMemCounter.checkOutstandingNewEq(0));

#if !defined(TEST_HAS_NO_EXCEPTIONS) && !defined(DISABLE_NEW_COUNT)
    {
        std::auto_ptr<A> ptr(new A);
        A* raw_ptr = ptr.get();
        globalMemCounter.throw_after = 0;
        try
        {
            std::shared_ptr<B> p(std::move(ptr));
            assert(false);
        }
        catch (...)
        {
            assert(A::count == 1);
            assert(B::count == 1);
            assert(ptr.get() == raw_ptr);
        }
    }
    assert(A::count == 0);
    assert(globalMemCounter.checkOutstandingNewEq(0));
#endif // !defined(TEST_HAS_NO_EXCEPTIONS) && !defined(DISABLE_NEW_COUNT)

  return 0;
}
