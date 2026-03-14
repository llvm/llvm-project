//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check that the nested types of std::allocator<void> are provided.
// After C++17, those are not provided in the primary template and the
// explicit specialization doesn't exist anymore, so this test is moot.

// REQUIRES: c++03 || c++11 || c++14 || c++17

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

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

#include <memory>
#include <type_traits>

static_assert((std::is_same<std::allocator<void>::pointer, void*>::value), "");
static_assert((std::is_same<std::allocator<void>::const_pointer, const void*>::value), "");
static_assert((std::is_same<std::allocator<void>::value_type, void>::value), "");
static_assert((std::is_same<std::allocator<void>::rebind<int>::other,
                            std::allocator<int> >::value), "");
