//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// template<class In>
// concept input_or_output_iterator;

#include <iterator>

#include "test_iterators.h"

static_assert(std::input_or_output_iterator<int*>);
static_assert(std::input_or_output_iterator<int const*>);
static_assert(std::input_or_output_iterator<int volatile*>);
static_assert(std::input_or_output_iterator<int const volatile*>);

static_assert(std::input_or_output_iterator<cpp17_input_iterator<int*> >);
static_assert(std::input_or_output_iterator<cpp17_input_iterator<int const*> >);
static_assert(std::input_or_output_iterator<cpp17_input_iterator<int volatile*> >);
static_assert(std::input_or_output_iterator<cpp17_input_iterator<int const volatile*> >);

static_assert(std::input_or_output_iterator<forward_iterator<int*> >);
static_assert(std::input_or_output_iterator<forward_iterator<int const*> >);
static_assert(std::input_or_output_iterator<forward_iterator<int volatile*> >);
static_assert(std::input_or_output_iterator<forward_iterator<int const volatile*> >);

static_assert(std::input_or_output_iterator<bidirectional_iterator<int*> >);
static_assert(std::input_or_output_iterator<bidirectional_iterator<int const*> >);
static_assert(std::input_or_output_iterator<bidirectional_iterator<int volatile*> >);
static_assert(std::input_or_output_iterator<bidirectional_iterator<int const volatile*> >);

static_assert(std::input_or_output_iterator<random_access_iterator<int*> >);
static_assert(std::input_or_output_iterator<random_access_iterator<int const*> >);
static_assert(std::input_or_output_iterator<random_access_iterator<int volatile*> >);
static_assert(std::input_or_output_iterator<random_access_iterator<int const volatile*> >);

static_assert(!std::input_or_output_iterator<void*>);
static_assert(!std::input_or_output_iterator<void const*>);
static_assert(!std::input_or_output_iterator<void volatile*>);
static_assert(!std::input_or_output_iterator<void const volatile*>);

struct S {};
static_assert(!std::input_or_output_iterator<S>);
static_assert(!std::input_or_output_iterator<int S::*>);
static_assert(!std::input_or_output_iterator<int (S::*)()>);
static_assert(!std::input_or_output_iterator<int (S::*)() const>);
static_assert(!std::input_or_output_iterator<int (S::*)() volatile>);

struct missing_dereference {
  using difference_type = std::ptrdiff_t;

  missing_dereference& operator++();
  missing_dereference& operator++(int);
};
static_assert(std::weakly_incrementable<missing_dereference> && !std::input_or_output_iterator<missing_dereference>);

struct void_dereference {
  using difference_type = std::ptrdiff_t;

  void operator*();
  void_dereference& operator++();
  void_dereference& operator++(int);
};
static_assert(std::weakly_incrementable<void_dereference> && !std::input_or_output_iterator<void_dereference>);

struct not_weakly_incrementable {
  int operator*() const;
};
static_assert(!std::input_or_output_iterator<not_weakly_incrementable>);
