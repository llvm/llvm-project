//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// Iterator traits and member typedefs in join_view::<iterator>.

#include <ranges>

#include "test_iterators.h"
#include "test_macros.h"
#include "../types.h"

template<class T>
struct ForwardView : std::ranges::view_base {
  forward_iterator<T*> begin() const;
  sentinel_wrapper<forward_iterator<T*>> end() const;
};

template<class T>
struct InputView : std::ranges::view_base {
  cpp17_input_iterator<T*> begin() const;
  sentinel_wrapper<cpp17_input_iterator<T*>> end() const;
};

template<class T>
concept HasIterCategory = requires { typename T::iterator_category; };

void test() {
  {
    int buffer[4][4];
    std::ranges::join_view jv(buffer);
    using Iter = std::ranges::iterator_t<decltype(jv)>;

    ASSERT_SAME_TYPE(Iter::iterator_concept, std::bidirectional_iterator_tag);
    ASSERT_SAME_TYPE(Iter::iterator_category, std::bidirectional_iterator_tag);
    ASSERT_SAME_TYPE(Iter::difference_type, std::ptrdiff_t);
    ASSERT_SAME_TYPE(Iter::value_type, int);
  }
  {
    using Iter = std::ranges::iterator_t<std::ranges::join_view<ForwardView<ForwardView<int>>>>;

    ASSERT_SAME_TYPE(Iter::iterator_concept, std::forward_iterator_tag);
    ASSERT_SAME_TYPE(Iter::iterator_category, std::forward_iterator_tag);
    ASSERT_SAME_TYPE(Iter::difference_type, std::ptrdiff_t);
    ASSERT_SAME_TYPE(Iter::value_type, int);
  }
  {
    using Iter = std::ranges::iterator_t<std::ranges::join_view<InputView<InputView<int>>>>;

    ASSERT_SAME_TYPE(Iter::iterator_concept, std::input_iterator_tag);
    static_assert(!HasIterCategory<Iter>);
    ASSERT_SAME_TYPE(Iter::difference_type, std::ptrdiff_t);
    ASSERT_SAME_TYPE(Iter::value_type, int);
  }
}
