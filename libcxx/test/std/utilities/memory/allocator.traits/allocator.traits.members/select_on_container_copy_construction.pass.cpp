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
//     static allocator_type
//         select_on_container_copy_construction(const allocator_type& a);
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
    {
        typedef IncompleteHolder* VT;
        typedef A<VT> Alloc;
        Alloc a;
        assert(std::allocator_traits<Alloc>::select_on_container_copy_construction(a).id == 0);
    }
#if TEST_STD_VER >= 11
    {
        B<int> b;
        assert(std::allocator_traits<B<int> >::select_on_container_copy_construction(b).id == 100);
    }
    {
        const B<int> b(0);
        assert(std::allocator_traits<B<int> >::select_on_container_copy_construction(b).id == 100);
    }
#endif
}
