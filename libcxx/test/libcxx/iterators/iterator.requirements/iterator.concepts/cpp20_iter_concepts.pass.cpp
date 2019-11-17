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

#include "test_macros.h"

#include <iterator>
struct OtherTag : std::input_iterator_tag {};
struct OtherTagTwo : std::output_iterator_tag {};
struct MyIter : std::iterator<std::random_access_iterator_tag, char> {
  using iterator_concept = int;
};
struct MyIter2 : std::iterator<OtherTag, char> {

};
struct MyIter3 {};

struct Empty {};
struct EmptyWithSpecial {};
namespace std {
template <>
struct iterator_traits<MyIter3>
    : std::iterator<OtherTagTwo, char> {};

template <>
struct iterator_traits<EmptyWithSpecial> {
  // empty non-default.
};
} // namespace std

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
