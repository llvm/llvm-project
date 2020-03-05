//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// check nested types:

// template <class T>
// class allocator
// {
// public:
//     typedef size_t                                size_type;
//     typedef ptrdiff_t                             difference_type;
//     typedef T*                                    pointer;
//     typedef const T*                              const_pointer;
//     typedef typename add_lvalue_reference<T>::type       reference;
//     typedef typename add_lvalue_reference<const T>::type const_reference;
//
//     template <class U> struct rebind {typedef allocator<U> other;};
// ...
// };

// Deprecated in C++17

// UNSUPPORTED: c++98, c++03, c++11, c++14
// REQUIRES: verify-support

// Clang 6 does not handle the deprecated attribute on template members properly,
// so the rebind<int> check below fails.
// UNSUPPORTED: clang-6

// MODULES_DEFINES: _LIBCPP_ENABLE_CXX20_REMOVED_ALLOCATOR_MEMBERS
#define _LIBCPP_ENABLE_CXX20_REMOVED_ALLOCATOR_MEMBERS

#include <memory>
#include "test_macros.h"

int main(int, char**)
{
    typedef std::allocator<char>::size_type AST;          // expected-error{{'size_type' is deprecated}}
    typedef std::allocator<char>::difference_type ADT;    // expected-error{{'difference_type' is deprecated}}
    typedef std::allocator<char>::pointer AP;             // expected-error{{'pointer' is deprecated}}
    typedef std::allocator<char>::const_pointer ACP;      // expected-error{{'const_pointer' is deprecated}}
    typedef std::allocator<char>::reference AR;           // expected-error{{'reference' is deprecated}}
    typedef std::allocator<char>::const_reference ACR;    // expected-error{{'const_reference' is deprecated}}
    typedef std::allocator<char>::rebind<int>::other ARO; // expected-error{{'rebind<int>' is deprecated}}

  return 0;
}
