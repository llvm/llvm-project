//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// template <class _Ptr, class _Tag, size_t _RangeCapacity>
// class __static_packed_bounded_iter;

#include <__iterator/static_packed_bounded_iter.h>
#include <cstddef>
#include <type_traits>

#include "test_iterators.h"

template <typename Ptr, std::size_t _Cap>
concept test =
    requires(Ptr a) { typename std::__static_packed_bounded_iterator<Ptr, std::decay_t<decltype(*a)>[], _Cap>; };

static_assert(test<std::int32_t*, 1>);
static_assert(test<std::int32_t*, 2>);
static_assert(test<std::int64_t*, 6>);

static_assert(!test<std::int8_t*, 1>);
static_assert(!test<std::int16_t*, 1>);
static_assert(!test<std::int32_t*, 3>);
static_assert(!test<std::int64_t*, 7>);

// passing non-pointer types
static_assert(!test<cpp20_random_access_iterator<int*>, 0>);
static_assert(!test<contiguous_iterator<int*>, 0>);
