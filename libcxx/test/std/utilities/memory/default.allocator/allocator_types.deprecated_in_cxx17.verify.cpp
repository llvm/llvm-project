//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// Check that the following nested types are deprecated in C++17:

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

// REQUIRES: c++17

// Clang 6 does not handle the deprecated attribute on template members properly,
// so the rebind<int> check below fails.
// UNSUPPORTED: clang-6

#include <memory>
#include "test_macros.h"

int main(int, char**)
{
    typedef std::allocator<char>::pointer AP;             // expected-warning {{'pointer' is deprecated}}
    typedef std::allocator<char>::const_pointer ACP;      // expected-warning {{'const_pointer' is deprecated}}
    typedef std::allocator<char>::reference AR;           // expected-warning {{'reference' is deprecated}}
    typedef std::allocator<char>::const_reference ACR;    // expected-warning {{'const_reference' is deprecated}}
    typedef std::allocator<char>::rebind<int>::other ARO; // expected-warning {{'rebind<int>' is deprecated}}

    typedef std::allocator<char const>::pointer AP2;             // expected-warning {{'pointer' is deprecated}}
    typedef std::allocator<char const>::const_pointer ACP2;      // expected-warning {{'const_pointer' is deprecated}}
    typedef std::allocator<char const>::reference AR2;           // expected-warning {{'reference' is deprecated}}
    typedef std::allocator<char const>::const_reference ACR2;    // expected-warning {{'const_reference' is deprecated}}
    typedef std::allocator<char const>::rebind<int>::other ARO2; // expected-warning {{'rebind<int>' is deprecated}}
    return 0;
}
