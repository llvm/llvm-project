//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>
//
// template <>
// class allocator<void>
// {
// public:
//     typedef void*                                 pointer;
//     typedef const void*                           const_pointer;
//     typedef void                                  value_type;
//
//     template <class _Up> struct rebind {typedef allocator<_Up> other;};
// };
//
// Deprecated in C++17

// UNSUPPORTED: c++98, c++03, c++11, c++14
// REQUIRES: verify-support

// MODULES_DEFINES: _LIBCPP_ENABLE_CXX20_REMOVED_ALLOCATOR_MEMBERS
#define _LIBCPP_ENABLE_CXX20_REMOVED_ALLOCATOR_MEMBERS

#include <memory>
#include "test_macros.h"

int main(int, char**)
{
    typedef std::allocator<void>::pointer AP;             // expected-error{{'allocator<void>' is deprecated}}
    typedef std::allocator<void>::const_pointer ACP;      // expected-error{{'allocator<void>' is deprecated}}
    typedef std::allocator<void>::rebind<int>::other ARO; // expected-error{{'allocator<void>' is deprecated}}

  return 0;
}
