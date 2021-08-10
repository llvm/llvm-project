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

// template<input_iterator I>
//   requires same_as<ITER_TRAITS(I), iterator_traits<I>>   // see [iterator.concepts.general]
// struct iterator_traits<counted_iterator<I>> : iterator_traits<I> {
//   using pointer = conditional_t<contiguous_iterator<I>,
//                                 add_pointer_t<iter_reference_t<I>>, void>;
// };

#include <iterator>

#include "test_macros.h"
#include "test_iterators.h"

void test() {
  {
    using Iter = cpp17_input_iterator<int*>;
    using CommonIter = std::counted_iterator<Iter>;
    using IterTraits = std::iterator_traits<CommonIter>;

    static_assert(std::same_as<IterTraits::iterator_category, std::input_iterator_tag>);
    static_assert(std::same_as<IterTraits::value_type, int>);
    static_assert(std::same_as<IterTraits::difference_type, std::ptrdiff_t>);
    static_assert(std::same_as<IterTraits::pointer, void>);
    static_assert(std::same_as<IterTraits::reference, int&>);
  }
  {
    using Iter = forward_iterator<int*>;
    using CommonIter = std::counted_iterator<Iter>;
    using IterTraits = std::iterator_traits<CommonIter>;

    static_assert(std::same_as<IterTraits::iterator_category, std::forward_iterator_tag>);
    static_assert(std::same_as<IterTraits::value_type, int>);
    static_assert(std::same_as<IterTraits::difference_type, std::ptrdiff_t>);
    static_assert(std::same_as<IterTraits::pointer, void>);
    static_assert(std::same_as<IterTraits::reference, int&>);
  }
  {
    using Iter = random_access_iterator<int*>;
    using CommonIter = std::counted_iterator<Iter>;
    using IterTraits = std::iterator_traits<CommonIter>;

    static_assert(std::same_as<IterTraits::iterator_category, std::random_access_iterator_tag>);
    static_assert(std::same_as<IterTraits::value_type, int>);
    static_assert(std::same_as<IterTraits::difference_type, std::ptrdiff_t>);
    static_assert(std::same_as<IterTraits::pointer, void>);
    static_assert(std::same_as<IterTraits::reference, int&>);
  }
  {
    using Iter = contiguous_iterator<int*>;
    using CommonIter = std::counted_iterator<Iter>;
    using IterTraits = std::iterator_traits<CommonIter>;

    static_assert(std::same_as<IterTraits::iterator_category, std::contiguous_iterator_tag>);
    static_assert(std::same_as<IterTraits::value_type, int>);
    static_assert(std::same_as<IterTraits::difference_type, std::ptrdiff_t>);
    static_assert(std::same_as<IterTraits::pointer, int*>);
    static_assert(std::same_as<IterTraits::reference, int&>);
  }
}
