//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class Alloc>
// struct allocator_traits
// {
//     template <class Ptr>
//         static void destroy(allocator_type& a, Ptr p);
//     ...
// };

#include <memory>
#include <new>
#include <type_traits>
#include <cassert>

template <class T>
struct A
{
    typedef T value_type;

};

int b_destroy = 0;

template <class T>
struct B
{
    typedef T value_type;

    template <class U>
    void destroy(U* p)
    {
        ++b_destroy;
        p->~U();
    }
};

struct A0
{
    static int count;
    ~A0() {++count;}
};

int A0::count = 0;

int main()
{
    {
        A0::count = 0;
        A<int> a;
        std::aligned_storage<sizeof(A0)>::type a0;
        std::allocator_traits<A<int> >::construct(a, (A0*)&a0);
        assert(A0::count == 0);
        std::allocator_traits<A<int> >::destroy(a, (A0*)&a0);
        assert(A0::count == 1);
    }
#ifndef _LIBCPP_HAS_NO_ADVANCED_SFINAE
    {
        A0::count = 0;
        b_destroy = 0;
        B<int> b;
        std::aligned_storage<sizeof(A0)>::type a0;
        std::allocator_traits<B<int> >::construct(b, (A0*)&a0);
        assert(A0::count == 0);
        assert(b_destroy == 0);
        std::allocator_traits<B<int> >::destroy(b, (A0*)&a0);
        assert(A0::count == 1);
        assert(b_destroy == 1);
    }
#endif  // _LIBCPP_HAS_NO_ADVANCED_SFINAE
}
