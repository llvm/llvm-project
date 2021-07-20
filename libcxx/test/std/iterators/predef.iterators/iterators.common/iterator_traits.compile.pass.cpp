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

// template<input_iterator I, class S>
//   struct iterator_traits<common_iterator<I, S>>;

#include <iterator>

#include "test_macros.h"
#include "types.h"

void test() {
  {
    using Iter = simple_iterator<int*>;
    using CommonIter = std::common_iterator<Iter, sentinel_type<int*>>;
    using IterTraits = std::iterator_traits<CommonIter>;

    static_assert(std::same_as<IterTraits::iterator_concept, std::input_iterator_tag>);
    static_assert(std::same_as<IterTraits::iterator_category, std::input_iterator_tag>);
    static_assert(std::same_as<IterTraits::value_type, int>);
    static_assert(std::same_as<IterTraits::difference_type, std::ptrdiff_t>);
    static_assert(std::same_as<IterTraits::pointer, int*>);
    static_assert(std::same_as<IterTraits::reference, int&>);
  }
  {
    using Iter = value_iterator<int*>;
    using CommonIter = std::common_iterator<Iter, sentinel_type<int*>>;
    using IterTraits = std::iterator_traits<CommonIter>;

    static_assert(std::same_as<IterTraits::iterator_concept, std::input_iterator_tag>);
    static_assert(std::same_as<IterTraits::iterator_category, std::input_iterator_tag>);
    static_assert(std::same_as<IterTraits::value_type, int>);
    static_assert(std::same_as<IterTraits::difference_type, std::ptrdiff_t>);
    // Note: IterTraits::pointer == __proxy.
    static_assert(!std::same_as<IterTraits::pointer, int*>);
    static_assert(std::same_as<IterTraits::reference, int>);
  }
  {
    using Iter = non_const_deref_iterator<int*>;
    using CommonIter = std::common_iterator<Iter, sentinel_type<int*>>;
    using IterTraits = std::iterator_traits<CommonIter>;

    static_assert(std::same_as<IterTraits::iterator_concept, std::input_iterator_tag>);
    static_assert(std::same_as<IterTraits::iterator_category, std::input_iterator_tag>);
    static_assert(std::same_as<IterTraits::value_type, int>);
    static_assert(std::same_as<IterTraits::difference_type, std::ptrdiff_t>);
    static_assert(std::same_as<IterTraits::pointer, void>);
    static_assert(std::same_as<IterTraits::reference, int&>);
  }
  {
    using Iter = cpp17_input_iterator<int*>;
    using CommonIter = std::common_iterator<Iter, sentinel_type<int*>>;
    using IterTraits = std::iterator_traits<CommonIter>;

    static_assert(std::same_as<IterTraits::iterator_concept, std::input_iterator_tag>);
    static_assert(std::same_as<IterTraits::iterator_category, std::input_iterator_tag>);
    static_assert(std::same_as<IterTraits::value_type, int>);
    static_assert(std::same_as<IterTraits::difference_type, std::ptrdiff_t>);
    static_assert(std::same_as<IterTraits::pointer, const Iter&>);
    static_assert(std::same_as<IterTraits::reference, int&>);
  }
  {
    using Iter = forward_iterator<int*>;
    using CommonIter = std::common_iterator<Iter, sentinel_type<int*>>;
    using IterTraits = std::iterator_traits<CommonIter>;

    static_assert(std::same_as<IterTraits::iterator_concept, std::forward_iterator_tag>);
    static_assert(std::same_as<IterTraits::iterator_category, std::forward_iterator_tag>);
    static_assert(std::same_as<IterTraits::value_type, int>);
    static_assert(std::same_as<IterTraits::difference_type, std::ptrdiff_t>);
    static_assert(std::same_as<IterTraits::pointer, const Iter&>);
    static_assert(std::same_as<IterTraits::reference, int&>);
  }
  {
    using Iter = random_access_iterator<int*>;
    using CommonIter = std::common_iterator<Iter, sentinel_type<int*>>;
    using IterTraits = std::iterator_traits<CommonIter>;

    static_assert(std::same_as<IterTraits::iterator_concept, std::forward_iterator_tag>);
    static_assert(std::same_as<IterTraits::iterator_category, std::forward_iterator_tag>);
    static_assert(std::same_as<IterTraits::value_type, int>);
    static_assert(std::same_as<IterTraits::difference_type, std::ptrdiff_t>);
    static_assert(std::same_as<IterTraits::pointer, const Iter&>);
    static_assert(std::same_as<IterTraits::reference, int&>);
  }

  // Testing iterator conformance.
  {
    static_assert(std::input_iterator<std::common_iterator<cpp17_input_iterator<int*>, sentinel_type<int*>>>);
    static_assert(std::forward_iterator<std::common_iterator<forward_iterator<int*>, sentinel_type<int*>>>);
    static_assert(std::forward_iterator<std::common_iterator<random_access_iterator<int*>, sentinel_type<int*>>>);
    static_assert(std::forward_iterator<std::common_iterator<contiguous_iterator<int*>, sentinel_type<int*>>>);
    // Even these are only forward.
    static_assert(!std::bidirectional_iterator<std::common_iterator<random_access_iterator<int*>, sentinel_type<int*>>>);
    static_assert(!std::bidirectional_iterator<std::common_iterator<contiguous_iterator<int*>, sentinel_type<int*>>>);

    using Iter = std::common_iterator<forward_iterator<int*>, sentinel_type<int*>>;
    static_assert(std::indirectly_writable<Iter, int>);
    static_assert(std::indirectly_swappable<Iter, Iter>);
  }
}
