//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// template <class _Iterator>
// struct __bounded_iter;
//
// Nested types

#include <__cxx03/__iterator/bounded_iter.h>
#include <cstddef>
#include <iterator>
#include <type_traits>

#include "test_macros.h"

using BoundedIter2 = std::__bounded_iter<int*>;
static_assert(std::is_same<BoundedIter2::value_type, int>::value, "");
static_assert(std::is_same<BoundedIter2::difference_type, std::ptrdiff_t>::value, "");
static_assert(std::is_same<BoundedIter2::pointer, int*>::value, "");
static_assert(std::is_same<BoundedIter2::reference, int&>::value, "");
static_assert(std::is_same<BoundedIter2::iterator_category, std::random_access_iterator_tag>::value, "");
