//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <generator>
// LWG 3900: allocator conversion is a Mandates requirement, even when an explicit conversion exists.

#include <generator>
#include <concepts>
#include <memory>

struct AllocatorArgument {
  explicit operator std::allocator<int>() const { return {}; }
};
static_assert(!std::convertible_to<const AllocatorArgument&, std::allocator<int>>);

std::generator<int, void, std::allocator<int>> generate(std::allocator_arg_t, AllocatorArgument) { co_yield 42; }
