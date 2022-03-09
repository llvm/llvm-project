//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// shared_ptr

// template<class Y> shared_ptr(const shared_ptr<Y>& r);

#include <memory>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

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

struct C
{
    static int count;

    C() {++count;}
    C(const C&) {++count;}
    virtual ~C() {--count;}
};

int C::count = 0;

class private_delete_op
{
    static void operator delete (void *p) {
        return delete static_cast<char*>(p);
    }
public:
    static void operator delete[] (void *p) {
        return delete[] static_cast<char*>(p);
    }
};

class private_delete_arr_op
{
    static void operator delete[] (void *p) {
        return delete[] static_cast<char*>(p);
    }
public:
    static void operator delete (void *p) {
        return delete static_cast<char*>(p);
    }
};

int main(int, char**)
{
    static_assert(( std::is_convertible<std::shared_ptr<A>, std::shared_ptr<B> >::value), "");
    static_assert((!std::is_convertible<std::shared_ptr<B>, std::shared_ptr<A> >::value), "");
    static_assert((!std::is_convertible<std::shared_ptr<A>, std::shared_ptr<C> >::value), "");
    {
        const std::shared_ptr<A> pA(new A);
        assert(pA.use_count() == 1);
        assert(B::count == 1);
        assert(A::count == 1);
        {
            std::shared_ptr<B> pB(pA);
            assert(B::count == 1);
            assert(A::count == 1);
            assert(pB.use_count() == 2);
            assert(pA.use_count() == 2);
            assert(pA.get() == pB.get());
        }
        assert(pA.use_count() == 1);
        assert(B::count == 1);
        assert(A::count == 1);
    }
    assert(B::count == 0);
    assert(A::count == 0);
    {
        std::shared_ptr<A> pA;
        assert(pA.use_count() == 0);
        assert(B::count == 0);
        assert(A::count == 0);
        {
            std::shared_ptr<B> pB(pA);
            assert(B::count == 0);
            assert(A::count == 0);
            assert(pB.use_count() == 0);
            assert(pA.use_count() == 0);
            assert(pA.get() == pB.get());
        }
        assert(pA.use_count() == 0);
        assert(B::count == 0);
        assert(A::count == 0);
    }
    assert(B::count == 0);
    assert(A::count == 0);

    // This should work in C++03 but we get errors when trying to do SFINAE with the delete operator.
    // GCC also complains about this.
#if TEST_STD_VER >= 11 && !defined(TEST_COMPILER_GCC)
    {
        // LWG2874: Make sure that when T (for std::shared_ptr<T>) is an array type,
        //          this constructor only participates in overload resolution when
        //          `delete[] p` is well formed. And when T is not an array type,
        //          this constructor only participates in overload resolution when
        //          `delete p` is well formed.
        static_assert(!std::is_constructible<std::shared_ptr<private_delete_op>,
                                                             private_delete_op*>::value, "");
        static_assert(!std::is_constructible<std::shared_ptr<private_delete_arr_op[4]>,
                                                             private_delete_arr_op*>::value, "");
    }
#endif

#if TEST_STD_VER > 14
    {
        std::shared_ptr<A[]> p1(new A[8]);
        assert(p1.use_count() == 1);
        assert(A::count == 8);
        {
            std::shared_ptr<const A[]> p2(p1);
            assert(A::count == 8);
            assert(p2.use_count() == 2);
            assert(p1.use_count() == 2);
            assert(p1.get() == p2.get());
        }
        assert(p1.use_count() == 1);
        assert(A::count == 8);
    }
    assert(A::count == 0);
#endif

    {
        std::shared_ptr<A const> pA(new A);
        std::shared_ptr<B const> pB(pA);
        assert(pB.get() == pA.get());
    }

    return 0;
}
