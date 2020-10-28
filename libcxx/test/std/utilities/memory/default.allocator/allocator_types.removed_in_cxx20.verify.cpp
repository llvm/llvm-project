//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// Check that the following nested types are removed in C++20:

// template <class T>
// class allocator
// {
// public:
//     typedef T*                                           pointer;
//     typedef const T*                                     const_pointer;
//     typedef typename add_lvalue_reference<T>::type       reference;
//     typedef typename add_lvalue_reference<const T>::type const_reference;
//
//     template <class U> struct rebind {typedef allocator<U> other;};
// ...
// };

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <memory>
#include "test_macros.h"

template <typename T>
void check()
{
    typedef typename std::allocator<T>::pointer AP;                      // expected-error 2 {{no type named 'pointer'}}
    typedef typename std::allocator<T>::const_pointer ACP;               // expected-error 2 {{no type named 'const_pointer'}}
    typedef typename std::allocator<T>::reference AR;                    // expected-error 2 {{no type named 'reference'}}
    typedef typename std::allocator<T>::const_reference ACR;             // expected-error 2 {{no type named 'const_reference'}}
    typedef typename std::allocator<T>::template rebind<int>::other ARO; // expected-error 2 {{no member named 'rebind'}}
}

int main(int, char**)
{
    check<char>();
    check<char const>();
    return 0;
}
