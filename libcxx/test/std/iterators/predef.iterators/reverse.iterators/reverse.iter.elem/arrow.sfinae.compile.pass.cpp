//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <iterator>
//
// reverse_iterator
//
// pointer operator->() const;

#include <iterator>

#include <type_traits>
#include "test_iterators.h"

template <class T>
concept HasArrow = requires(T t) { t.operator->(); };

struct simple_bidirectional_iterator {
    using iterator_category = std::bidirectional_iterator_tag;
    using value_type = int;
    using difference_type = int;
    using pointer = int*;
    using reference = int&;

    reference operator*() const;
    pointer operator->() const;

    simple_bidirectional_iterator& operator++();
    simple_bidirectional_iterator& operator--();
    simple_bidirectional_iterator operator++(int);
    simple_bidirectional_iterator operator--(int);

    friend bool operator==(const simple_bidirectional_iterator&, const simple_bidirectional_iterator&);
};
static_assert( std::bidirectional_iterator<simple_bidirectional_iterator>);
static_assert(!std::random_access_iterator<simple_bidirectional_iterator>);

using PtrRI = std::reverse_iterator<int*>;
static_assert( HasArrow<PtrRI>);

using PtrLikeRI = std::reverse_iterator<simple_bidirectional_iterator>;
static_assert( HasArrow<PtrLikeRI>);

// `bidirectional_iterator` from `test_iterators.h` doesn't define `operator->`.
using NonPtrRI = std::reverse_iterator<bidirectional_iterator<int*>>;
static_assert(!HasArrow<NonPtrRI>);
