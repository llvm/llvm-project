//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <functional>

// class function<R(ArgTypes...)>

// template <MoveConstructible  R, MoveConstructible ... ArgTypes>
//   void swap(function<R(ArgTypes...)>&, function<R(ArgTypes...)>&) noexcept;

#include <functional>
#include <cstdlib>
#include <cassert>

#include "test_macros.h"
#include "count_new.h"

class A
{
    int data_[10];
public:
    static int count;

    explicit A(int j)
    {
        ++count;
        data_[0] = j;
    }

    A(const A& a)
    {
        ++count;
        for (int i = 0; i < 10; ++i)
            data_[i] = a.data_[i];
    }

    ~A() {--count;}

    int operator()(int i) const
    {
        for (int j = 0; j < 10; ++j)
            i += data_[j];
        return i;
    }

    int id() const {return data_[0];}
};

int A::count = 0;

int g(int) {return 0;}
int h(int) {return 1;}

int main(int, char**)
{
    globalMemCounter.reset();
    assert(globalMemCounter.checkOutstandingNewEq(0));
    {
    std::function<int(int)> f1 = A(1);
    std::function<int(int)> f2 = A(2);
#if TEST_STD_VER >= 11
    static_assert(noexcept(swap(f1, f2)), "" );
#endif
    assert(A::count == 2);
    assert(globalMemCounter.checkOutstandingNewEq(2));
    RTTI_ASSERT(f1.target<A>()->id() == 1);
    RTTI_ASSERT(f2.target<A>()->id() == 2);
    swap(f1, f2);
    assert(A::count == 2);
    assert(globalMemCounter.checkOutstandingNewEq(2));
    RTTI_ASSERT(f1.target<A>()->id() == 2);
    RTTI_ASSERT(f2.target<A>()->id() == 1);
    }
    assert(A::count == 0);
    assert(globalMemCounter.checkOutstandingNewEq(0));
    {
    std::function<int(int)> f1 = A(1);
    std::function<int(int)> f2 = g;
#if TEST_STD_VER >= 11
    static_assert(noexcept(swap(f1, f2)), "" );
#endif
    assert(A::count == 1);
    assert(globalMemCounter.checkOutstandingNewEq(1));
    RTTI_ASSERT(f1.target<A>()->id() == 1);
    RTTI_ASSERT(*f2.target<int(*)(int)>() == g);
    swap(f1, f2);
    assert(A::count == 1);
    assert(globalMemCounter.checkOutstandingNewEq(1));
    RTTI_ASSERT(*f1.target<int(*)(int)>() == g);
    RTTI_ASSERT(f2.target<A>()->id() == 1);
    }
    assert(A::count == 0);
    assert(globalMemCounter.checkOutstandingNewEq(0));
    {
    std::function<int(int)> f1 = g;
    std::function<int(int)> f2 = A(1);
#if TEST_STD_VER >= 11
    static_assert(noexcept(swap(f1, f2)), "" );
#endif
    assert(A::count == 1);
    assert(globalMemCounter.checkOutstandingNewEq(1));
    RTTI_ASSERT(*f1.target<int(*)(int)>() == g);
    RTTI_ASSERT(f2.target<A>()->id() == 1);
    swap(f1, f2);
    assert(A::count == 1);
    assert(globalMemCounter.checkOutstandingNewEq(1));
    RTTI_ASSERT(f1.target<A>()->id() == 1);
    RTTI_ASSERT(*f2.target<int(*)(int)>() == g);
    }
    assert(A::count == 0);
    assert(globalMemCounter.checkOutstandingNewEq(0));
    {
    std::function<int(int)> f1 = g;
    std::function<int(int)> f2 = h;
#if TEST_STD_VER >= 11
    static_assert(noexcept(swap(f1, f2)), "" );
#endif
    assert(A::count == 0);
    assert(globalMemCounter.checkOutstandingNewEq(0));
    RTTI_ASSERT(*f1.target<int(*)(int)>() == g);
    RTTI_ASSERT(*f2.target<int(*)(int)>() == h);
    swap(f1, f2);
    assert(A::count == 0);
    assert(globalMemCounter.checkOutstandingNewEq(0));
    RTTI_ASSERT(*f1.target<int(*)(int)>() == h);
    RTTI_ASSERT(*f2.target<int(*)(int)>() == g);
    }
    assert(A::count == 0);
    assert(globalMemCounter.checkOutstandingNewEq(0));

  return 0;
}
