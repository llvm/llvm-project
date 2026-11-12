//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<class In, class Out>
// concept indirectly_movable;

#include <iterator>

#include "MoveOnly.h"
#include "test_macros.h"

// Can move between pointers.
static_assert( std::indirectly_movable<int*, int*>);
static_assert( std::indirectly_movable<const int*, int*>);
static_assert(!std::indirectly_movable<int*, const int*>);
static_assert( std::indirectly_movable<const int*, int*>);

// Can move from a pointer into an array but arrays aren't considered indirectly movable-from.
static_assert( std::indirectly_movable<int*, int[2]>);
static_assert(!std::indirectly_movable<int[2], int*>);
static_assert(!std::indirectly_movable<int[2], int[2]>);
static_assert(!std::indirectly_movable<int(&)[2], int(&)[2]>);

// Can't move between non-pointer types.
static_assert(!std::indirectly_movable<int*, int>);
static_assert(!std::indirectly_movable<int, int*>);
static_assert(!std::indirectly_movable<int, int>);

// Check some less common types.
static_assert(!std::indirectly_movable<void*, void*>);
static_assert(!std::indirectly_movable<int*, void*>);
static_assert(!std::indirectly_movable<int(), int()>);
static_assert(!std::indirectly_movable<int*, int()>);
static_assert(!std::indirectly_movable<void, void>);

// Can move move-only objects.
static_assert( std::indirectly_movable<MoveOnly*, MoveOnly*>);
static_assert(!std::indirectly_movable<MoveOnly*, const MoveOnly*>);
static_assert(!std::indirectly_movable<const MoveOnly*, const MoveOnly*>);
static_assert(!std::indirectly_movable<const MoveOnly*, MoveOnly*>);

template<class T>
struct PointerTo {
  using value_type = T;
  T& operator*() const;
};

// Can copy through a dereferenceable class.
static_assert( std::indirectly_movable<int*, PointerTo<int>>);
static_assert(!std::indirectly_movable<int*, PointerTo<const int>>);
static_assert( std::indirectly_copyable<PointerTo<int>, PointerTo<int>>);
static_assert(!std::indirectly_copyable<PointerTo<int>, PointerTo<const int>>);
static_assert( std::indirectly_movable<MoveOnly*, PointerTo<MoveOnly>>);
static_assert( std::indirectly_movable<PointerTo<MoveOnly>, MoveOnly*>);
static_assert( std::indirectly_movable<PointerTo<MoveOnly>, PointerTo<MoveOnly>>);
