//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <vector>

#include <type_traits>
#include <vector>

#include "test_allocator.h"

namespace std_allocator {
using vec = std::vector<int>;

static_assert(std::is_nothrow_constructible<vec>::value);
static_assert(std::is_nothrow_constructible<vec, const std::allocator<int>&>::value);
static_assert(std::is_nothrow_constructible<vec, vec&&>::value);

static_assert(std::is_nothrow_assignable<vec&, vec&&>::value);
static_assert(std::is_nothrow_swappable<vec>::value);
} // namespace std_allocator

namespace test_alloc {
using vec = std::vector<int, test_allocator<int>>;

static_assert(std::is_nothrow_constructible<vec>::value);
static_assert(std::is_nothrow_constructible<vec, const test_allocator<int>&>::value);
static_assert(std::is_nothrow_constructible<vec, vec&&>::value);

static_assert(!std::is_nothrow_assignable<vec&, vec&&>::value);
static_assert(std::is_nothrow_swappable<vec>::value);
} // namespace test_alloc
