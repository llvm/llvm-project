//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

//  using iterator_category = If<
//    derived_from<typename iterator_traits<iterator_t<Base>>::iterator_category, forward_iterator_tag>,
//    forward_iterator_tag,
//    typename iterator_traits<iterator_t<Base>>::iterator_category
//  >;
//  using iterator_concept = typename outer-iterator<Const>::iterator_concept;
//  using value_type = range_value_t<Base>;
//  using difference_type = range_difference_t<Base>;

#include <ranges>

#include <concepts>
#include <iterator>
#include "../types.h"

template <class Range, class Pattern>
using OuterIter = std::ranges::iterator_t<std::ranges::lazy_split_view<Range, Pattern>>;
template <class Range, class Pattern>
using InnerIter = std::ranges::iterator_t<decltype(*OuterIter<Range, Pattern>())>;

// iterator_concept

static_assert(std::same_as<typename InnerIter<ForwardView, ForwardView>::iterator_concept,
    typename OuterIter<ForwardView, ForwardView>::iterator_concept>);
static_assert(std::same_as<typename InnerIter<InputView, ForwardTinyView>::iterator_concept,
    typename OuterIter<InputView, ForwardTinyView>::iterator_concept>);

// iterator_category

static_assert(std::same_as<typename InnerIter<ForwardView, ForwardView>::iterator_category, std::forward_iterator_tag>);

template <class Range, class Pattern>
concept NoIteratorCategory = !requires { typename InnerIter<Range, Pattern>::iterator_category; };
static_assert(NoIteratorCategory<InputView, ForwardTinyView>);

// value_type

static_assert(std::same_as<typename InnerIter<ForwardView, ForwardView>::value_type,
    std::ranges::range_value_t<ForwardView>>);

// difference_type

static_assert(std::same_as<typename InnerIter<ForwardView, ForwardView>::difference_type,
    std::ranges::range_difference_t<ForwardView>>);
