//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// std::forward_iterator;

#include <iterator>

#include <concepts>

#include "test_iterators.h"

static_assert(!std::forward_iterator<cpp17_input_iterator<int*> >);
static_assert(!std::forward_iterator<cpp20_input_iterator<int*> >);
static_assert(std::forward_iterator<forward_iterator<int*> >);
static_assert(std::forward_iterator<bidirectional_iterator<int*> >);
static_assert(std::forward_iterator<random_access_iterator<int*> >);
static_assert(std::forward_iterator<contiguous_iterator<int*> >);

static_assert(std::forward_iterator<int*>);
static_assert(std::forward_iterator<int const*>);
static_assert(std::forward_iterator<int volatile*>);
static_assert(std::forward_iterator<int const volatile*>);

struct not_input_iterator {
  // using value_type = int;
  using difference_type = std::ptrdiff_t;
  using iterator_concept = std::forward_iterator_tag;

  int operator*() const;

  not_input_iterator& operator++();
  not_input_iterator operator++(int);

  bool operator==(not_input_iterator const&) const = default;
};
static_assert(std::input_or_output_iterator<not_input_iterator>);
static_assert(!std::input_iterator<not_input_iterator>);
static_assert(!std::forward_iterator<not_input_iterator>);

struct bad_iterator_tag {
  using value_type = int;
  using difference_type = std::ptrdiff_t;
  using iterator_concept = std::input_iterator_tag;

  int operator*() const;

  bad_iterator_tag& operator++();
  bad_iterator_tag operator++(int);

  bool operator==(bad_iterator_tag const&) const = default;
};
static_assert(!std::forward_iterator<bad_iterator_tag>);

struct not_incrementable {
  using value_type = int;
  using difference_type = std::ptrdiff_t;
  using iterator_concept = std::forward_iterator_tag;

  int operator*() const;

  not_incrementable& operator++();
  void operator++(int);

  bool operator==(not_incrementable const&) const = default;
};
static_assert(!std::forward_iterator<not_incrementable>);

struct not_equality_comparable {
  using value_type = int;
  using difference_type = std::ptrdiff_t;
  using iterator_concept = std::forward_iterator_tag;

  int operator*() const;

  not_equality_comparable& operator++();
  not_equality_comparable operator++(int);

  bool operator==(not_equality_comparable const&) const = delete;
};
static_assert(!std::forward_iterator<not_equality_comparable>);
