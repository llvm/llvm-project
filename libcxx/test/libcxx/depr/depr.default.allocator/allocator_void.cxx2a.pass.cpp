//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// Check that the following member types of allocator<void> are provided
// regardless of the Standard when we request them from libc++.

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

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_CXX20_REMOVED_ALLOCATOR_MEMBERS
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

#include <memory>
#include <type_traits>

#include "test_macros.h"

int main(int, char**)
{
    static_assert((std::is_same<std::allocator<void>::pointer, void*>::value), "");
    static_assert((std::is_same<std::allocator<void>::const_pointer, const void*>::value), "");
    static_assert((std::is_same<std::allocator<void>::value_type, void>::value), "");
    static_assert((std::is_same<std::allocator<void>::rebind<int>::other,
                                std::allocator<int> >::value), "");
    std::allocator<void> a;
    std::allocator<void> a2 = a;
    a2 = a;

  return 0;
}
