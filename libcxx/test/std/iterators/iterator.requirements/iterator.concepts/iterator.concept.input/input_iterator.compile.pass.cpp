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

// template<class T>
// concept input_iterator;

#include <iterator>

#include "test_iterators.h"

static_assert(std::input_iterator<cpp17_input_iterator<int*> >);
static_assert(std::input_iterator<cpp20_input_iterator<int*> >);

struct no_explicit_iter_concept {
  using value_type = int;
  using difference_type = std::ptrdiff_t;

  no_explicit_iter_concept() = default;

  no_explicit_iter_concept(no_explicit_iter_concept&&) = default;
  no_explicit_iter_concept& operator=(no_explicit_iter_concept&&) = default;

  no_explicit_iter_concept(no_explicit_iter_concept const&) = delete;
  no_explicit_iter_concept& operator=(no_explicit_iter_concept const&) = delete;

  value_type operator*() const;

  no_explicit_iter_concept& operator++();
  void operator++(int);
};
// ITER-CONCEPT is `random_access_iterator_tag` >:(
static_assert(std::input_iterator<no_explicit_iter_concept>);

static_assert(std::input_iterator<int*>);
static_assert(std::input_iterator<int const*>);
static_assert(std::input_iterator<int volatile*>);
static_assert(std::input_iterator<int const volatile*>);

struct not_weakly_incrementable {
  using difference_type = std::ptrdiff_t;
  using iterator_concept = std::input_iterator_tag;

  not_weakly_incrementable() = default;

  not_weakly_incrementable(not_weakly_incrementable&&) = default;
  not_weakly_incrementable& operator=(not_weakly_incrementable&&) = default;

  not_weakly_incrementable(not_weakly_incrementable const&) = delete;
  not_weakly_incrementable& operator=(not_weakly_incrementable const&) = delete;

  int operator*() const;

  not_weakly_incrementable& operator++();
};
static_assert(!std::input_or_output_iterator<not_weakly_incrementable> &&
              !std::input_iterator<not_weakly_incrementable>);

struct not_indirectly_readable {
  using difference_type = std::ptrdiff_t;
  using iterator_concept = std::input_iterator_tag;

  not_indirectly_readable() = default;

  not_indirectly_readable(not_indirectly_readable&&) = default;
  not_indirectly_readable& operator=(not_indirectly_readable&&) = default;

  not_indirectly_readable(not_indirectly_readable const&) = delete;
  not_indirectly_readable& operator=(not_indirectly_readable const&) = delete;

  int operator*() const;

  not_indirectly_readable& operator++();
  void operator++(int);
};
static_assert(!std::indirectly_readable<not_indirectly_readable> && !std::input_iterator<not_indirectly_readable>);

struct bad_iterator_category {
  using value_type = int;
  using difference_type = std::ptrdiff_t;
  using iterator_category = void;

  bad_iterator_category() = default;

  bad_iterator_category(bad_iterator_category&&) = default;
  bad_iterator_category& operator=(bad_iterator_category&&) = default;

  bad_iterator_category(bad_iterator_category const&) = delete;
  bad_iterator_category& operator=(bad_iterator_category const&) = delete;

  value_type operator*() const;

  bad_iterator_category& operator++();
  void operator++(int);
};
static_assert(!std::input_iterator<bad_iterator_category>);

struct bad_iterator_concept {
  using value_type = int;
  using difference_type = std::ptrdiff_t;
  using iterator_concept = void*;

  bad_iterator_concept() = default;

  bad_iterator_concept(bad_iterator_concept&&) = default;
  bad_iterator_concept& operator=(bad_iterator_concept&&) = default;

  bad_iterator_concept(bad_iterator_concept const&) = delete;
  bad_iterator_concept& operator=(bad_iterator_concept const&) = delete;

  value_type operator*() const;

  bad_iterator_concept& operator++();
  void operator++(int);
};
static_assert(!std::input_iterator<bad_iterator_concept>);
