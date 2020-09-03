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
//     static pointer allocate(allocator_type& a, size_type n);
//     ...
// };

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <memory>
#include <cstdint>
#include <cassert>

#include "test_macros.h"

template <class T>
struct A
{
    typedef T value_type;

    value_type* allocate(std::size_t n)
    {
        assert(n == 12);
        return reinterpret_cast<value_type*>(static_cast<std::uintptr_t>(0xEEADBEEF));
    }
    value_type* allocate(std::size_t n, const void* p)
    {
        assert(n == 11);
        assert(p == 0);
        return reinterpret_cast<value_type*>(static_cast<std::uintptr_t>(0xFEADBEEF));
    }
};

int main(int, char**)
{
    A<int> a;
    std::allocator_traits<A<int> >::allocate(a, 10);          // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::allocator_traits<A<int> >::allocate(a, 10, nullptr); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    return 0;
}
