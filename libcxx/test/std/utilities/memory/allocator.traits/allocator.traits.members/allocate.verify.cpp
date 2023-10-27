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
//     static constexpr pointer allocate(allocator_type& a, size_type n);
//     ...
// };

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <cstddef>
#include <memory>

template <class T>
struct A {
    typedef T value_type;
    value_type* allocate(std::size_t n);
    value_type* allocate(std::size_t n, const void* p);
};

void f() {
    A<int> a;
    std::allocator_traits<A<int> >::allocate(a, 10);          // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::allocator_traits<A<int> >::allocate(a, 10, nullptr); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
