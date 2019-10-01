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
//     static size_type max_size(const allocator_type& a) noexcept;
//     ...
// };

#include <limits>
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

    size_t max_size() const
    {
        return 100;
    }
};

int main(int, char**)
{
    {
        B<int> b;
        assert(std::allocator_traits<B<int> >::max_size(b) == 100);
    }
    {
        const B<int> b = {};
        assert(std::allocator_traits<B<int> >::max_size(b) == 100);
    }
    {
        typedef IncompleteHolder* VT;
        typedef B<VT> Alloc;
        Alloc a;
        assert(std::allocator_traits<Alloc >::max_size(a) == 100);
    }
#if TEST_STD_VER >= 11
    {
        A<int> a;
        assert(std::allocator_traits<A<int> >::max_size(a) ==
               std::numeric_limits<std::size_t>::max() / sizeof(int));
    }
    {
        const A<int> a = {};
        assert(std::allocator_traits<A<int> >::max_size(a) ==
               std::numeric_limits<std::size_t>::max() / sizeof(int));
    }
    {
        std::allocator<int> a;
        static_assert(noexcept(std::allocator_traits<std::allocator<int>>::max_size(a)) == true, "");
    }
#endif

  return 0;
}
