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

#include <cstddef>
#include <iterator>
#include <type_traits>

#include "test_macros.h"

#if TEST_STD_VER > 17
struct Iterator {
  struct value_type {};
  using difference_type = int;
  struct pointer {};
  using reference = value_type&;
  struct iterator_category : std::random_access_iterator_tag {};
  using iterator_concept = std::contiguous_iterator_tag;
};

using BoundedIter1 = std::__bounded_iter<Iterator>;
static_assert(std::is_same<BoundedIter1::value_type, Iterator::value_type>::value, "");
static_assert(std::is_same<BoundedIter1::difference_type, Iterator::difference_type>::value, "");
static_assert(std::is_same<BoundedIter1::pointer, Iterator::pointer>::value, "");
static_assert(std::is_same<BoundedIter1::reference, Iterator::reference>::value, "");
static_assert(std::is_same<BoundedIter1::iterator_category, Iterator::iterator_category>::value, "");
static_assert(std::is_same<BoundedIter1::iterator_concept, Iterator::iterator_concept>::value, "");
#endif


using BoundedIter2 = std::__bounded_iter<int*>;
static_assert(std::is_same<BoundedIter2::value_type, int>::value, "");
static_assert(std::is_same<BoundedIter2::difference_type, std::ptrdiff_t>::value, "");
static_assert(std::is_same<BoundedIter2::pointer, int*>::value, "");
static_assert(std::is_same<BoundedIter2::reference, int&>::value, "");
static_assert(std::is_same<BoundedIter2::iterator_category, std::random_access_iterator_tag>::value, "");
#if TEST_STD_VER > 17
static_assert(std::is_same<BoundedIter2::iterator_concept, std::contiguous_iterator_tag>::value, "");
#endif
