//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// Member typedefs in elements_view<V, N>::iterator.

#include <concepts>
#include <ranges>
#include <tuple>

#include "test_macros.h"
#include "test_iterators.h"

template <class Iter>
using Range = std::ranges::subrange<Iter, sentinel_wrapper<Iter>>;

template <class Range, size_t N = 0>
using ElementsIter = std::ranges::iterator_t<std::ranges::elements_view<Range, N>>;

// using iterator_concept = see below;
static_assert(std::same_as<ElementsIter<Range<cpp20_input_iterator<std::tuple<int>*>>>::iterator_concept,
                           std::input_iterator_tag>);

static_assert(std::same_as<ElementsIter<Range<forward_iterator<std::tuple<int>*>>>::iterator_concept, //
                           std::forward_iterator_tag>);

static_assert(std::same_as<ElementsIter<Range<bidirectional_iterator<std::tuple<int>*>>>::iterator_concept,
                           std::bidirectional_iterator_tag>);

static_assert(std::same_as<ElementsIter<Range<random_access_iterator<std::tuple<int>*>>>::iterator_concept,
                           std::random_access_iterator_tag>);

static_assert(std::same_as<ElementsIter<Range<contiguous_iterator<std::tuple<int>*>>>::iterator_concept,
                           std::random_access_iterator_tag>);

static_assert(std::same_as<ElementsIter<Range<std::tuple<int>*>>::iterator_concept, //
                           std::random_access_iterator_tag>);

// using iterator_category = see below;   // not always present
template <class T>
concept HasIterCategory = requires { typename T::iterator_category; };

static_assert(!HasIterCategory<ElementsIter<Range<cpp20_input_iterator<std::tuple<int>*>>>>);
static_assert(HasIterCategory<ElementsIter<Range<forward_iterator<std::tuple<int>*>>>>);

static_assert(std::same_as<ElementsIter<Range<forward_iterator<std::tuple<int>*>>>::iterator_category,
                           std::forward_iterator_tag>);

static_assert(std::same_as<ElementsIter<Range<bidirectional_iterator<std::tuple<int>*>>>::iterator_category,
                           std::bidirectional_iterator_tag>);

static_assert(std::same_as<ElementsIter<Range<random_access_iterator<std::tuple<int>*>>>::iterator_category,
                           std::random_access_iterator_tag>);

static_assert(std::same_as<ElementsIter<Range<contiguous_iterator<std::tuple<int>*>>>::iterator_category,
                           std::random_access_iterator_tag>);

static_assert(std::same_as<ElementsIter<Range<std::tuple<int>*>>::iterator_category, //
                           std::random_access_iterator_tag>);

using Generator = decltype(std::views::iota(0, 1) | std::views::transform([](int) {
                             return std::pair<int, short>{1, 1};
                           }));
static_assert(std::ranges::random_access_range<Generator>);

static_assert(std::same_as<ElementsIter<Generator>::iterator_category, //
                           std::input_iterator_tag>);

// using value_type = remove_cvref_t<tuple_element_t<N, range_value_t<Base>>>;
static_assert(std::same_as<ElementsIter<Range<std::tuple<int, long>*>, 0>::value_type, int>);

static_assert(std::same_as<ElementsIter<Range<std::tuple<int, long>*>, 1>::value_type, long>);

static_assert(std::same_as<ElementsIter<Generator, 0>::value_type, int>);

static_assert(std::same_as<ElementsIter<Generator, 1>::value_type, short>);

// using difference_type = range_difference_t<Base>;
static_assert(std::same_as<ElementsIter<Range<std::tuple<int, long>*>>::difference_type,
                           std::ranges::range_difference_t<Range<std::tuple<int, long>*>>>);

static_assert(std::same_as<ElementsIter<Generator>::difference_type, //
                           std::ranges::range_difference_t<Generator>>);
