//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <memory>

// shared_ptr

// shared_ptr(shared_ptr&& r);

#include <memory>
#include <cassert>

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
    {
        std::shared_ptr<A> pA(new A);
        assert(pA.use_count() == 1);
        assert(A::count == 1);
        {
            A* p = pA.get();
            std::shared_ptr<A> pA2(std::move(pA));
            assert(A::count == 1);
#if TEST_STD_VER >= 11
            assert(pA.use_count() == 0);
            assert(pA2.use_count() == 1);
#else
            assert(pA.use_count() == 2);
            assert(pA2.use_count() == 2);
#endif
            assert(pA2.get() == p);
        }
#if TEST_STD_VER >= 11
        assert(pA.use_count() == 0);
        assert(A::count == 0);
#else
        assert(pA.use_count() == 1);
        assert(A::count == 1);
#endif
    }

    assert(A::count == 0);
    {
        std::shared_ptr<A> pA;
        assert(pA.use_count() == 0);
        assert(A::count == 0);
        {
            std::shared_ptr<A> pA2(std::move(pA));
            assert(A::count == 0);
            assert(pA.use_count() == 0);
            assert(pA2.use_count() == 0);
            assert(pA2.get() == pA.get());
        }
        assert(pA.use_count() == 0);
        assert(A::count == 0);
    }
    assert(A::count == 0);

    {
        std::shared_ptr<A const> pA(new A);
        A const* p = pA.get();
        std::shared_ptr<A const> pA2(std::move(pA));
        assert(pA2.get() == p);
    }

  return 0;
}
