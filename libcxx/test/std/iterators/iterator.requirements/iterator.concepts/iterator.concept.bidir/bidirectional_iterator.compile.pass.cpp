//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// template<class T>
// concept bidirectional_iterator;

#include <iterator>

#include <concepts>

#include "test_iterators.h"

static_assert(!std::bidirectional_iterator<cpp17_input_iterator<int*> >);
static_assert(!std::bidirectional_iterator<cpp20_input_iterator<int*> >);
static_assert(!std::bidirectional_iterator<forward_iterator<int*> >);
static_assert(std::bidirectional_iterator<bidirectional_iterator<int*> >);
static_assert(std::bidirectional_iterator<random_access_iterator<int*> >);
static_assert(std::bidirectional_iterator<contiguous_iterator<int*> >);


static_assert(std::bidirectional_iterator<int*>);
static_assert(std::bidirectional_iterator<int const*>);
static_assert(std::bidirectional_iterator<int volatile*>);
static_assert(std::bidirectional_iterator<int const volatile*>);

struct not_forward_iterator {
  using value_type = int;
  using difference_type = std::ptrdiff_t;
  using iterator_concept = std::bidirectional_iterator_tag;

  value_type operator*() const;

  not_forward_iterator& operator++();
  not_forward_iterator operator++(int);

  not_forward_iterator& operator--();
  not_forward_iterator& operator--(int);
};
static_assert(std::input_iterator<not_forward_iterator> && !std::forward_iterator<not_forward_iterator> &&
              !std::bidirectional_iterator<not_forward_iterator>);

struct wrong_iterator_category {
  using value_type = int;
  using difference_type = std::ptrdiff_t;
  using iterator_category = std::forward_iterator_tag;

  value_type& operator*() const;

  wrong_iterator_category& operator++();
  wrong_iterator_category operator++(int);

  wrong_iterator_category& operator--();
  wrong_iterator_category operator--(int);

  bool operator==(wrong_iterator_category const&) const = default;
};
static_assert(!std::bidirectional_iterator<wrong_iterator_category>);

struct wrong_iterator_concept {
  using value_type = int;
  using difference_type = std::ptrdiff_t;
  using iterator_concept = std::forward_iterator_tag;

  value_type& operator*() const;

  wrong_iterator_concept& operator++();
  wrong_iterator_concept operator++(int);

  wrong_iterator_concept& operator--();
  wrong_iterator_concept operator--(int);

  bool operator==(wrong_iterator_concept const&) const = default;
};
static_assert(!std::bidirectional_iterator<wrong_iterator_concept>);

struct no_predecrement {
  using value_type = int;
  using difference_type = std::ptrdiff_t;
  using iterator_concept = std::bidirectional_iterator_tag;

  value_type& operator*() const;

  no_predecrement& operator++();
  no_predecrement operator++(int);

  no_predecrement operator--(int);

  bool operator==(no_predecrement const&) const = default;
};
static_assert(!std::bidirectional_iterator<no_predecrement>);

struct bad_predecrement {
  using value_type = int;
  using difference_type = std::ptrdiff_t;
  using iterator_concept = std::bidirectional_iterator_tag;

  value_type& operator*() const;

  bad_predecrement& operator++();
  bad_predecrement operator++(int);

  bad_predecrement operator--();
  bad_predecrement operator--(int);

  bool operator==(bad_predecrement const&) const = default;
};
static_assert(!std::bidirectional_iterator<bad_predecrement>);

struct no_postdecrement {
  using value_type = int;
  using difference_type = std::ptrdiff_t;
  using iterator_concept = std::bidirectional_iterator_tag;

  value_type& operator*() const;

  no_postdecrement& operator++();
  no_postdecrement operator++(int);

  no_postdecrement& operator--();

  bool operator==(no_postdecrement const&) const = default;
};
static_assert(!std::bidirectional_iterator<no_postdecrement>);

struct bad_postdecrement {
  using value_type = int;
  using difference_type = std::ptrdiff_t;
  using iterator_concept = std::bidirectional_iterator_tag;

  value_type& operator*() const;

  bad_postdecrement& operator++();
  bad_postdecrement operator++(int);

  bad_postdecrement& operator--();
  bad_postdecrement& operator--(int);

  bool operator==(bad_postdecrement const&) const = default;
};
static_assert(!std::bidirectional_iterator<bad_postdecrement>);
