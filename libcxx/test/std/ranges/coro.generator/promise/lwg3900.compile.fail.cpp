//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <generator>
// LWG 3900: an incompatible allocator argument must not cause fallback to default allocation.

#include <generator>
#include <memory>
#include <memory_resource>

std::pmr::generator<int> generate(std::allocator_arg_t, std::pmr::memory_resource&) { co_yield 42; }
