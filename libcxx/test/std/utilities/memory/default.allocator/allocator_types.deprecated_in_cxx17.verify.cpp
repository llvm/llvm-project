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

#include <memory>

void f() {
    {
        typedef std::allocator<char>::pointer Pointer;                  // expected-warning {{'pointer' is deprecated}}
        typedef std::allocator<char>::const_pointer ConstPointer;       // expected-warning {{'const_pointer' is deprecated}}
        typedef std::allocator<char>::reference Reference;              // expected-warning {{'reference' is deprecated}}
        typedef std::allocator<char>::const_reference ConstReference;   // expected-warning {{'const_reference' is deprecated}}
        typedef std::allocator<char>::rebind<int>::other Rebind;        // expected-warning {{'rebind<int>' is deprecated}}
    }
    {
        typedef std::allocator<void>::pointer Pointer;                  // expected-warning {{'pointer' is deprecated}}
        typedef std::allocator<void>::const_pointer ConstPointer;       // expected-warning {{'const_pointer' is deprecated}}
        // reference and const_reference are not provided by std::allocator<void>
        typedef std::allocator<void>::rebind<int>::other Rebind;        // expected-warning {{'rebind<int>' is deprecated}}
    }
}
