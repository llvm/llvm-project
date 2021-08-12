//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// template<class In, class Out>
// concept indirectly_movable;

#include <iterator>

#include "test_macros.h"

struct IndirectlyMovableWithInt {
  int& operator*() const;
};

struct Empty {};

struct MoveOnly {
  MoveOnly(MoveOnly&&) = default;
  MoveOnly(MoveOnly const&) = delete;
  MoveOnly& operator=(MoveOnly&&) = default;
  MoveOnly& operator=(MoveOnly const&) = delete;
  MoveOnly() = default;
};

template<class T>
struct PointerTo {
  using value_type = T;
  T& operator*() const;
};

static_assert( std::indirectly_movable<int*, int*>);
static_assert( std::indirectly_movable<const int*, int *>);
static_assert(!std::indirectly_movable<int*, const int *>);
static_assert(!std::indirectly_movable<const int*, const int *>);
static_assert( std::indirectly_movable<int*, int[2]>);
static_assert(!std::indirectly_movable<int[2], int*>);
static_assert(!std::indirectly_movable<int[2], int[2]>);
static_assert(!std::indirectly_movable<int(&)[2], int(&)[2]>);
static_assert(!std::indirectly_movable<int, int*>);
static_assert(!std::indirectly_movable<int, int>);
static_assert( std::indirectly_movable<Empty*, Empty*>);
static_assert( std::indirectly_movable<int*, IndirectlyMovableWithInt>);
static_assert(!std::indirectly_movable<Empty*, IndirectlyMovableWithInt>);
static_assert( std::indirectly_movable<int*, IndirectlyMovableWithInt>);
static_assert( std::indirectly_movable<MoveOnly*, MoveOnly*>);
static_assert(!std::indirectly_movable<MoveOnly*, const MoveOnly*>);
static_assert(!std::indirectly_movable<const MoveOnly*, const MoveOnly*>);
static_assert(!std::indirectly_movable<const MoveOnly*, MoveOnly*>);
static_assert( std::indirectly_movable<PointerTo<MoveOnly>, PointerTo<MoveOnly>>);
static_assert( std::indirectly_movable<MoveOnly*, PointerTo<MoveOnly>>);
