//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <deque>

// deque(deque&&)
//        noexcept(is_nothrow_move_constructible<allocator_type>::value);

// This tests a conforming extension

// UNSUPPORTED: c++03

#include <cassert>
#include <deque>

#include "MoveOnly.h"
#include "test_allocator.h"

template <class T>
struct some_alloc {
  typedef T value_type;
  some_alloc(const some_alloc&);
  void allocate(std::size_t);
};

static_assert(std::is_nothrow_move_constructible<std::deque<MoveOnly>>::value, "");
static_assert(std::is_nothrow_move_constructible<std::deque<MoveOnly, test_allocator<MoveOnly>>>::value, "");
static_assert(std::is_nothrow_move_constructible<std::deque<MoveOnly, other_allocator<MoveOnly>>>::value, "");
static_assert(std::is_nothrow_move_constructible<std::deque<MoveOnly, some_alloc<MoveOnly>>>::value, "");
