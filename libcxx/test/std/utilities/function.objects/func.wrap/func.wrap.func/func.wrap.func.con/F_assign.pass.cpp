//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>

// class function<R(ArgTypes...)>

// template<class F>
//   requires CopyConstructible<F> && Callable<F, ArgTypes..>
//         && Convertible<Callable<F, ArgTypes...>::result_type
//   operator=(F f);

// This test runs in C++03, but we have deprecated using std::function in C++03.
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS -D_LIBCPP_ENABLE_CXX03_FUNCTION

#include <functional>
#include <cassert>

#include "test_macros.h"
#include "count_new.h"

class A
{
    int data_[10];
public:
    static int count;

    A()
    {
        ++count;
        for (int i = 0; i < 10; ++i)
            data_[i] = i;
    }

    A(const A&) {++count;}

    ~A() {--count;}

    int operator()(int i) const
    {
        for (int j = 0; j < 10; ++j)
            i += data_[j];
        return i;
    }

    int foo(int) const {return 1;}
};

int A::count = 0;

int g(int) {return 0;}

#if TEST_STD_VER >= 11
struct RValueCallable {
    template <class ...Args>
    void operator()(Args&&...) && {}
};
struct LValueCallable {
    template <class ...Args>
    void operator()(Args&&...) & {}
};
#endif

int main(int, char**)
{
    globalMemCounter.reset();
    assert(globalMemCounter.checkOutstandingNewEq(0));
    {
    std::function<int(int)> f;
    f = A();
    assert(A::count == 1);
    assert(globalMemCounter.checkOutstandingNewEq(1));
    RTTI_ASSERT(f.target<A>());
    RTTI_ASSERT(f.target<int(*)(int)>() == 0);
    }
    assert(A::count == 0);
    assert(globalMemCounter.checkOutstandingNewEq(0));
    {
    std::function<int(int)> f;
    f = g;
    assert(globalMemCounter.checkOutstandingNewEq(0));
    RTTI_ASSERT(f.target<int(*)(int)>());
    RTTI_ASSERT(f.target<A>() == 0);
    }
    assert(globalMemCounter.checkOutstandingNewEq(0));
    {
    std::function<int(int)> f;
    f = (int (*)(int))0;
    assert(!f);
    assert(globalMemCounter.checkOutstandingNewEq(0));
    RTTI_ASSERT(f.target<int(*)(int)>() == 0);
    RTTI_ASSERT(f.target<A>() == 0);
    }
    {
    std::function<int(const A*, int)> f;
    f = &A::foo;
    assert(f);
    assert(globalMemCounter.checkOutstandingNewEq(0));
    RTTI_ASSERT(f.target<int (A::*)(int) const>() != 0);
    }
    {
    std::function<void(int)> f;
    f = &g;
    assert(f);
    RTTI_ASSERT(f.target<int(*)(int)>() != 0);
    f(1);
    }
#if TEST_STD_VER >= 11
    {
        using Fn = std::function<void(int, int, int)>;
        static_assert(std::is_assignable<Fn&, LValueCallable&>::value, "");
        static_assert(std::is_assignable<Fn&, LValueCallable>::value, "");
        static_assert(!std::is_assignable<Fn&, RValueCallable&>::value, "");
        static_assert(!std::is_assignable<Fn&, RValueCallable>::value, "");
    }
    {
        using Fn = std::function<void(int, int, int)>;
        static_assert(std::is_assignable<Fn&, Fn&&>::value, "");
    }
    {
        using F1 = std::function<void(int, int)>;
        using F2 = std::function<void(int, int, int)>;
        static_assert(!std::is_assignable<F1&, F2&&>::value, "");
    }
    {
        using F1 = std::function<int(int, int)>;
        using F2 = std::function<A  (int, int)>;
        static_assert(!std::is_assignable<F1&, F2&&>::value, "");
        static_assert(!std::is_assignable<F2&, F1&&>::value, "");
    }
#endif

  return 0;
}
