//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// template <class _Iterator, class _Container, class _ContainerMaxElements>
// struct __capacity_aware_iterator;

// Nested types

#include "test_iterators.h"
#include <__iterator/capacity_aware_iterator.h>
#include <iterator>
#include <type_traits>

using It = contiguous_iterator<int*>;

using CapIter = std::__capacity_aware_iterator<It, int[], 1>;

static_assert(std::is_same_v<CapIter::iterator_category, It::iterator_category>);
static_assert(std::is_same_v<CapIter::iterator_concept, std::contiguous_iterator_tag>);
static_assert(std::is_same_v<CapIter::difference_type, std::iter_difference_t<It>>);
static_assert(std::is_same_v<CapIter::reference, std::iter_reference_t<It>>);
static_assert(std::is_same_v<CapIter::reference, std::iter_reference_t<It>>);
static_assert(std::is_same_v<CapIter::value_type, std::iter_value_t<It>>);
