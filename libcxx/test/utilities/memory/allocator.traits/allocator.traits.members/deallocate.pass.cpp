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
//     static void deallocate(allocator_type& a, pointer p, size_type n);
//     ...
// };

#include <memory>
#include <cassert>

int called = 0;

template <class T>
struct A
{
    typedef T value_type;

    void deallocate(value_type* p, std::size_t n)
    {
        assert(p == (value_type*)0xDEADBEEF);
        assert(n == 10);
        ++called;
    }
};

int main()
{
    A<int> a;
    std::allocator_traits<A<int> >::deallocate(a, (int*)0xDEADBEEF, 10);
    assert(called == 1);
}
