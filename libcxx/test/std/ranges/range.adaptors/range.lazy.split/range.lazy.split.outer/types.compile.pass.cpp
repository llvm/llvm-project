//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// using iterator_category = input_iterator_tag; // Only defined if `View` is a forward range.
// using iterator_concept = conditional_t<forward_range<Base>, forward_iterator_tag, input_iterator_tag>;
// using difference_type = range_difference_t<Base>;

#include <ranges>

#include <concepts>
#include <iterator>
#include "../types.h"

template <class Range, class Pattern>
using OuterIter = decltype(std::declval<std::ranges::lazy_split_view<Range, Pattern>>().begin());

// iterator_category

static_assert(std::same_as<typename OuterIter<ForwardView, ForwardView>::iterator_category, std::input_iterator_tag>);

template <class Range, class Pattern>
concept NoIteratorCategory = !requires { typename OuterIter<Range, Pattern>::iterator_category; };
static_assert(NoIteratorCategory<InputView, ForwardTinyView>);

// iterator_concept

static_assert(std::same_as<typename OuterIter<ForwardView, ForwardView>::iterator_concept, std::forward_iterator_tag>);
static_assert(std::same_as<typename OuterIter<InputView, ForwardTinyView>::iterator_concept, std::input_iterator_tag>);

// difference_type

static_assert(std::same_as<typename OuterIter<ForwardView, ForwardView>::difference_type,
    std::ranges::range_difference_t<ForwardView>>);
