//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: clang-8
// UNSUPPORTED: gcc-8, gcc-9

// <memory>

// template <class Alloc>
// struct allocator_traits
// {
//     static constexpr pointer allocate(allocator_type& a, size_type n, const_void_pointer hint);
//     ...
// };

#include <memory>
#include <cstdint>
#include <cassert>

#include "test_macros.h"
#include "incomplete_type_helper.h"

template <class T>
struct A
{
    typedef T value_type;

    TEST_CONSTEXPR_CXX20 A() {}

    TEST_CONSTEXPR_CXX20 value_type* allocate(std::size_t n)
    {
        assert(n == 10);
        return &storage;
    }

    value_type storage;
};

template <class T>
struct B
{
    typedef T value_type;

    TEST_CONSTEXPR_CXX20 value_type* allocate(std::size_t n, const void* p)
    {
        assert(n == 11);
        assert(p == nullptr);
        return &storage;
    }

    value_type storage;
};

TEST_CONSTEXPR_CXX20 bool test()
{
#if TEST_STD_VER >= 11
    {
        A<int> a;
        assert(std::allocator_traits<A<int> >::allocate(a, 10, nullptr) == &a.storage);
    }
    {
        typedef A<IncompleteHolder*> Alloc;
        Alloc a;
        assert(std::allocator_traits<Alloc>::allocate(a, 10, nullptr) == &a.storage);
    }
#endif
    {
        B<int> b;
        assert(std::allocator_traits<B<int> >::allocate(b, 11, nullptr) == &b.storage);
    }
    {
        typedef B<IncompleteHolder*> Alloc;
        Alloc b;
        assert(std::allocator_traits<Alloc>::allocate(b, 11, nullptr) == &b.storage);
    }

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
