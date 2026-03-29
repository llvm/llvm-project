//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// ITER_TRAITS(I)

// -- If the qualified-id ITER_TRAITS(I)::iterator_concept is valid and names a
//   type, then ITER_CONCEPT(I) denotes that type.
// (1.2) -- Otherwise, if the qualified-id ITER_TRAITS(I)::iterator_category is
// valid and names a type, then ITER_CONCEPT(I) denotes that type.
// (1.3) -- Otherwise, if iterator_traits<I> names a specialization generated
// from the primary template, then ITER_CONCEPT(I) denotes
// random_access_iterator_tag.
// (1.4) -- Otherwise, ITER_CONCEPT(I) does not denote a type.

#include <__type_traits/is_valid_expansion.h>
#include <cstddef>
#include <iterator>

#include "test_macros.h"

struct OtherTag : std::input_iterator_tag {};
struct OtherTagTwo : std::output_iterator_tag {};

struct MyIter {
  using iterator_category = std::random_access_iterator_tag;
  using iterator_concept = int;
  using value_type = char;
  using difference_type = std::ptrdiff_t;
  using pointer = char*;
  using reference = char&;
};

struct MyIter2 {
  using iterator_category = OtherTag;
  using value_type = char;
  using difference_type = std::ptrdiff_t;
  using pointer = char*;
  using reference = char&;
};

struct MyIter3 {};

struct Empty {};
struct EmptyWithSpecial {};
template <>
struct std::iterator_traits<MyIter3> {
  using iterator_category = OtherTagTwo;
  using value_type = char;
  using difference_type = std::ptrdiff_t;
  using pointer = char*;
  using reference = char&;
};

template <>
struct std::iterator_traits<EmptyWithSpecial> {
  // empty non-default.
};

int main(int, char**) {
  // If the qualified-id ITER_TRAITS(I)::iterator_concept is valid and names a type,
  // then ITER_CONCEPT(I) denotes that type.
  {
#if TEST_STD_VER > 17
    ASSERT_SAME_TYPE(std::_ITER_CONCEPT<char*>, std::contiguous_iterator_tag);
#endif
    ASSERT_SAME_TYPE(std::_ITER_CONCEPT<MyIter>, int);
  }
  // Otherwise, if the qualified-id ITER_TRAITS(I)::iterator_category is valid
  // and names a type, then ITER_CONCEPT(I) denotes that type.
  {
    ASSERT_SAME_TYPE(std::_ITER_CONCEPT<MyIter2>, OtherTag);
    ASSERT_SAME_TYPE(std::_ITER_CONCEPT<MyIter3>, OtherTagTwo);
  }
  // FIXME - This requirement makes no sense to me. Why does an empty type with
  // an empty default iterator_traits get a category of random?
  {
    ASSERT_SAME_TYPE(std::_ITER_CONCEPT<Empty>, std::random_access_iterator_tag);
  }
  {
    static_assert(!std::_IsValidExpansion<std::_ITER_CONCEPT, EmptyWithSpecial>::value, "");
  }

  return 0;
}
