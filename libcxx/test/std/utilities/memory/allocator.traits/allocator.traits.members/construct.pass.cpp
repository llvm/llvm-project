//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class Alloc>
// struct allocator_traits
// {
//     template <class Ptr, class... Args>
//     static constexpr void construct(allocator_type& a, Ptr p, Args&&... args);
//     ...
// };

#include <memory>
#include <new>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "incomplete_type_helper.h"

template <class T>
struct A
{
    typedef T value_type;

};

template <class T>
struct B
{
    typedef T value_type;

    TEST_CONSTEXPR_CXX20 B(int& count) : count(count) {}

#if TEST_STD_VER >= 11
    template <class U, class ...Args>
    TEST_CONSTEXPR_CXX20 void construct(U* p, Args&& ...args)
    {
        ++count;
#if TEST_STD_VER > 17
        std::construct_at(p, std::forward<Args>(args)...);
#else
        ::new ((void*)p) U(std::forward<Args>(args)...);
#endif
    }
#endif

    int& count;
};

struct A0
{
    TEST_CONSTEXPR_CXX20 A0(int* count) {++*count;}
};

struct A1
{
    TEST_CONSTEXPR_CXX20 A1(int* count, char c)
    {
        assert(c == 'c');
        ++*count;
    }
};

struct A2
{
    TEST_CONSTEXPR_CXX20 A2(int* count, char c, int i)
    {
        assert(c == 'd');
        assert(i == 5);
        ++*count;
    }
};

TEST_CONSTEXPR_CXX20 bool test()
{
    {
        int A0_count = 0;
        A<A0> a;
        std::allocator<A0> alloc;
        A0* a0 = alloc.allocate(1);
        assert(A0_count == 0);
        std::allocator_traits<A<A0> >::construct(a, a0, &A0_count);
        assert(A0_count == 1);
        alloc.deallocate(a0, 1);
    }
    {
        int A1_count = 0;
        A<A1> a;
        std::allocator<A1> alloc;
        A1* a1 = alloc.allocate(1);
        assert(A1_count == 0);
        std::allocator_traits<A<A1> >::construct(a, a1, &A1_count, 'c');
        assert(A1_count == 1);
        alloc.deallocate(a1, 1);
    }
    {
        int A2_count = 0;
        A<A2> a;
        std::allocator<A2> alloc;
        A2* a2 = alloc.allocate(1);
        assert(A2_count == 0);
        std::allocator_traits<A<A2> >::construct(a, a2, &A2_count, 'd', 5);
        assert(A2_count == 1);
        alloc.deallocate(a2, 1);
    }
    {
      typedef IncompleteHolder* VT;
      typedef A<VT> Alloc;
      Alloc a;
      std::allocator<VT> alloc;
      VT* vt = alloc.allocate(1);
      std::allocator_traits<Alloc>::construct(a, vt, nullptr);
      alloc.deallocate(vt, 1);
    }

#if TEST_STD_VER >= 11
    {
        int A0_count = 0;
        int b_construct = 0;
        B<A0> b(b_construct);
        std::allocator<A0> alloc;
        A0* a0 = alloc.allocate(1);
        assert(A0_count == 0);
        assert(b_construct == 0);
        std::allocator_traits<B<A0> >::construct(b, a0, &A0_count);
        assert(A0_count == 1);
        assert(b_construct == 1);
        alloc.deallocate(a0, 1);
    }
    {
        int A1_count = 0;
        int b_construct = 0;
        B<A1> b(b_construct);
        std::allocator<A1> alloc;
        A1* a1 = alloc.allocate(1);
        assert(A1_count == 0);
        assert(b_construct == 0);
        std::allocator_traits<B<A1> >::construct(b, a1, &A1_count, 'c');
        assert(A1_count == 1);
        assert(b_construct == 1);
        alloc.deallocate(a1, 1);
    }
    {
        int A2_count = 0;
        int b_construct = 0;
        B<A2> b(b_construct);
        std::allocator<A2> alloc;
        A2* a2 = alloc.allocate(1);
        assert(A2_count == 0);
        assert(b_construct == 0);
        std::allocator_traits<B<A2> >::construct(b, a2, &A2_count, 'd', 5);
        assert(A2_count == 1);
        assert(b_construct == 1);
        alloc.deallocate(a2, 1);
    }
#endif

    return true;
}

int main(int, char**)
{
    test();

#if TEST_STD_VER > 17
    static_assert(test());
#endif

    return 0;
}
