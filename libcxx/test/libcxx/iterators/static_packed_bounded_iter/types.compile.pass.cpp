//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// template <class _Ptr, class _Tag, class _RangeCapacity>
// class __packed_static_bounded_iterator;
//
// Nested types

#include <__iterator/static_packed_bounded_iter.h>
#include <cstddef>
#include <iterator>
#include <type_traits>

using Iter = std::__static_packed_bounded_iterator<int*, int[], 2>;

static_assert(std::is_same_v<Iter::value_type, int>, "");
static_assert(std::is_same_v<Iter::difference_type, std::ptrdiff_t>, "");
static_assert(std::is_same_v<Iter::pointer, int*>, "");
static_assert(std::is_same_v<Iter::reference, int&>, "");
static_assert(std::is_same_v<Iter::iterator_category, std::random_access_iterator_tag>, "");
static_assert(std::is_same_v<Iter::iterator_concept, std::contiguous_iterator_tag>, "");

static_assert(sizeof(Iter) == sizeof(void*));
