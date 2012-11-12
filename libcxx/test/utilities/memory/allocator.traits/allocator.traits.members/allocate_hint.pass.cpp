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
//     static pointer allocate(allocator_type& a, size_type n, const_void_pointer hint);
//     ...
// };

#include <memory>
#include <cassert>

template <class T>
struct A
{
    typedef T value_type;

    value_type* allocate(std::size_t n)
    {
        assert(n == 10);
        return (value_type*)0xDEADBEEF;
    }
};

template <class T>
struct B
{
    typedef T value_type;

    value_type* allocate(std::size_t n)
    {
        assert(n == 12);
        return (value_type*)0xEEADBEEF;
    }
    value_type* allocate(std::size_t n, const void* p)
    {
        assert(n == 11);
        assert(p == 0);
        return (value_type*)0xFEADBEEF;
    }
};

int main()
{
#ifndef _LIBCPP_HAS_NO_ADVANCED_SFINAE
    A<int> a;
    assert(std::allocator_traits<A<int> >::allocate(a, 10, nullptr) == (int*)0xDEADBEEF);
#endif  // _LIBCPP_HAS_NO_ADVANCED_SFINAE
    B<int> b;
    assert(std::allocator_traits<B<int> >::allocate(b, 11, nullptr) == (int*)0xFEADBEEF);
}
