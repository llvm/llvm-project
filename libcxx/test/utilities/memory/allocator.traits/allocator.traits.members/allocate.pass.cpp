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
//     static pointer allocate(allocator_type& a, size_type n);
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

int main()
{
    A<int> a;
    assert(std::allocator_traits<A<int> >::allocate(a, 10) == (int*)0xDEADBEEF);
}
