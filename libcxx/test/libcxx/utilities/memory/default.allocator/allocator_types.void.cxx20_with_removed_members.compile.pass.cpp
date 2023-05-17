//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check that the nested types of std::allocator<void> are provided in C++20
// with a flag that keeps the removed members.

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_CXX20_REMOVED_ALLOCATOR_MEMBERS
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_CXX20_REMOVED_ALLOCATOR_VOID_SPECIALIZATION

#include <memory>
#include <type_traits>

static_assert((std::is_same<std::allocator<void>::pointer, void*>::value), "");
static_assert((std::is_same<std::allocator<void>::const_pointer, const void*>::value), "");
static_assert((std::is_same<std::allocator<void>::value_type, void>::value), "");
static_assert((std::is_same<std::allocator<void>::rebind<int>::other, std::allocator<int> >::value), "");
