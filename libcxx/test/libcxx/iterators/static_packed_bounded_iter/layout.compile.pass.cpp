//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// template <class _Ptr, class _Tag, class _RangeCapacity>
// class __static_packed_bounded_iterator;
//
// Make sure that the this static_packed_bounded_iterator occupies as much memory as a pointer

#include <__iterator/static_packed_bounded_iter.h>
#include <cstddef>
#include <type_traits>

using Iter = std::__static_packed_bounded_iterator<int*, int[], 2>;

static_assert(sizeof(Iter) == sizeof(void*));
static_assert(alignof(Iter) == alignof(void*));
static_assert(std::is_trivially_copyable_v<Iter>);
static_assert(std::is_trivially_copy_assignable_v<Iter>);
static_assert(std::is_trivially_move_constructible_v<Iter>);
static_assert(std::is_trivially_move_assignable_v<Iter>);
static_assert(std::is_trivially_destructible_v<Iter>);
