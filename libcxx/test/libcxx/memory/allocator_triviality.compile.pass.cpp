//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

// Make sure that std::allocator<T> is trivial.

// <memory>

#include <memory>
#include <string>
#include <type_traits>

static_assert(std::is_trivially_default_constructible<std::allocator<char> >::value, "");
static_assert(std::is_trivially_default_constructible<std::allocator<std::string> >::value, "");
static_assert(std::is_trivially_default_constructible<std::allocator<void> >::value, "");

static_assert(std::is_trivially_copyable<std::allocator<char> >::value, "");
static_assert(std::is_trivially_copyable<std::allocator<std::string> >::value, "");
static_assert(std::is_trivially_copyable<std::allocator<void> >::value, "");
