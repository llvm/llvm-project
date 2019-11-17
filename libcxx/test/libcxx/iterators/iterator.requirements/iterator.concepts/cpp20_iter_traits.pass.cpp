//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// ITER_TRAITS(I)

// For a type I, let ITER_TRAITS(I) denote the type I if iterator_traits<I> names
// a specialization generated from the primary template. Otherwise,
// ITER_TRAITS(I) denotes iterator_traits<I>.

#include "test_macros.h"

#include <iterator>

struct MyIter : std::iterator<std::random_access_iterator_tag, char> {};
struct MyIter2 : std::iterator<std::random_access_iterator_tag, char> {};
struct MyIter3 : std::iterator<std::random_access_iterator_tag, char> {};

namespace std {
template <>
struct iterator_traits<MyIter>
    : iterator_traits<std::iterator<std::random_access_iterator_tag, char> > {};
template <>
struct iterator_traits<MyIter2>
    : std::iterator<std::random_access_iterator_tag, char> {};

} // namespace std

int main(int, char**) {
  ASSERT_SAME_TYPE(std::_ITER_TRAITS<char*>, std::iterator_traits<char*>);
  {
    using ClassIter = std::reverse_iterator<char*>;
    ASSERT_SAME_TYPE(std::_ITER_TRAITS<ClassIter>, ClassIter);
    ASSERT_SAME_TYPE(std::_ITER_TRAITS<MyIter3>, MyIter3);
  }
  {
    ASSERT_SAME_TYPE(std::_ITER_TRAITS<MyIter>, std::iterator_traits<MyIter>);
    ASSERT_SAME_TYPE(std::_ITER_TRAITS<MyIter2>, std::iterator_traits<MyIter2>);
  }
  return 0;
}
