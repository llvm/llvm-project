//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: gcc-10

// iterator_type, value_type, difference_type, iterator_concept, iterator_category

#include <iterator>

#include "test_macros.h"
#include "test_iterators.h"

// No value_type.
struct InputOrOutputArchetype {
  using difference_type = int;

  int operator*();
  void operator++(int);
  InputOrOutputArchetype& operator++();
};

template<class T>
concept HasValueType = requires { typename T::value_type; };

template<class T>
concept HasIteratorConcept = requires { typename T::iterator_concept; };

template<class T>
concept HasIteratorCategory = requires { typename T::iterator_category; };

void test() {
  {
    using Iter = std::counted_iterator<InputOrOutputArchetype>;
    static_assert(std::same_as<Iter::iterator_type, InputOrOutputArchetype>);
    static_assert(!HasValueType<Iter>);
    static_assert(std::same_as<Iter::difference_type, int>);
    static_assert(!HasIteratorConcept<Iter>);
    static_assert(!HasIteratorCategory<Iter>);
  }
  {
    using Iter = std::counted_iterator<cpp20_input_iterator<int*>>;
    static_assert(std::same_as<Iter::iterator_type, cpp20_input_iterator<int*>>);
    static_assert(std::same_as<Iter::value_type, int>);
    static_assert(std::same_as<Iter::difference_type, std::ptrdiff_t>);
    static_assert(std::same_as<Iter::iterator_concept, std::input_iterator_tag>);
    static_assert(!HasIteratorCategory<Iter>);
  }
  {
    using Iter = std::counted_iterator<random_access_iterator<int*>>;
    static_assert(std::same_as<Iter::iterator_type, random_access_iterator<int*>>);
    static_assert(std::same_as<Iter::value_type, int>);
    static_assert(std::same_as<Iter::difference_type, std::ptrdiff_t>);
    static_assert(!HasIteratorConcept<Iter>);
    static_assert(std::same_as<Iter::iterator_category, std::random_access_iterator_tag>);
  }
}
