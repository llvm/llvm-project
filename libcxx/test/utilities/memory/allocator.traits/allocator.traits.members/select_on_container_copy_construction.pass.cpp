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
//     static allocator_type
//         select_on_container_copy_construction(const allocator_type& a);
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
    int id;
    explicit A(int i = 0) : id(i) {}

};

template <class T>
struct B
{
    typedef T value_type;

    int id;
    explicit B(int i = 0) : id(i) {}

    B select_on_container_copy_construction() const
    {
        return B(100);
    }
};

int main()
{
    {
        A<int> a;
        assert(std::allocator_traits<A<int> >::select_on_container_copy_construction(a).id == 0);
    }
    {
        const A<int> a(0);
        assert(std::allocator_traits<A<int> >::select_on_container_copy_construction(a).id == 0);
    }
#ifndef _LIBCPP_HAS_NO_ADVANCED_SFINAE
    {
        B<int> b;
        assert(std::allocator_traits<B<int> >::select_on_container_copy_construction(b).id == 100);
    }
    {
        const B<int> b(0);
        assert(std::allocator_traits<B<int> >::select_on_container_copy_construction(b).id == 100);
    }
#endif  // _LIBCPP_HAS_NO_ADVANCED_SFINAE
}
