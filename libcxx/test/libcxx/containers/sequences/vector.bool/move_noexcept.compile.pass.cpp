//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// vector(vector&&)
//        noexcept(is_nothrow_move_constructible<allocator_type>::value);

// This tests a conforming extension

// UNSUPPORTED: c++03

#include <vector>
#include <cassert>

#include "test_allocator.h"

template <class T>
struct some_alloc {
  typedef T value_type;
  some_alloc(const some_alloc&);
};

static_assert(std::is_nothrow_move_constructible<std::vector<bool>>::value, "");
static_assert(std::is_nothrow_move_constructible<std::vector<bool, test_allocator<bool>>>::value, "");
static_assert(std::is_nothrow_move_constructible<std::vector<bool, other_allocator<bool>>>::value, "");
static_assert(std::is_nothrow_move_constructible<std::vector<bool, some_alloc<bool>>>::value, "");
