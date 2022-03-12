//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// template<class R>
// concept common_range;

#include <ranges>

#include "test_iterators.h"

template<class It>             struct Common { It begin() const; It end() const; };
template<class It>             struct NonCommon { It begin() const; sentinel_wrapper<It> end() const; };
template<class It, class Sent> struct Range { It begin() const; Sent end() const; };

static_assert(!std::ranges::common_range<Common<cpp17_input_iterator<int*>>>); // not a sentinel for itself
static_assert(!std::ranges::common_range<Common<cpp20_input_iterator<int*>>>); // not a sentinel for itself
static_assert( std::ranges::common_range<Common<forward_iterator<int*>>>);
static_assert( std::ranges::common_range<Common<bidirectional_iterator<int*>>>);
static_assert( std::ranges::common_range<Common<random_access_iterator<int*>>>);
static_assert( std::ranges::common_range<Common<contiguous_iterator<int*>>>);
static_assert( std::ranges::common_range<Common<int*>>);

static_assert(!std::ranges::common_range<NonCommon<cpp17_input_iterator<int*>>>);
static_assert(!std::ranges::common_range<NonCommon<cpp20_input_iterator<int*>>>);
static_assert(!std::ranges::common_range<NonCommon<forward_iterator<int*>>>);
static_assert(!std::ranges::common_range<NonCommon<bidirectional_iterator<int*>>>);
static_assert(!std::ranges::common_range<NonCommon<random_access_iterator<int*>>>);
static_assert(!std::ranges::common_range<NonCommon<contiguous_iterator<int*>>>);
static_assert(!std::ranges::common_range<NonCommon<int*>>);

// Test when begin() and end() only differ by their constness.
static_assert(!std::ranges::common_range<Range<int*, int const*>>);

// Simple test with a sized_sentinel.
static_assert(!std::ranges::common_range<Range<int*, sized_sentinel<int*>>>);

// Make sure cv-qualification doesn't impact the concept when begin() and end() have matching qualifiers.
static_assert( std::ranges::common_range<Common<forward_iterator<int*>> const>);
static_assert(!std::ranges::common_range<NonCommon<forward_iterator<int*>> const>);

// Test with a range that's a common_range only when const-qualified.
struct Range1 {
  int* begin();
  int const* begin() const;
  int const* end() const;
};
static_assert(!std::ranges::common_range<Range1>);
static_assert( std::ranges::common_range<Range1 const>);

// Test with a range that's a common_range only when not const-qualified.
struct Range2 {
  int* begin() const;
  int* end();
  int const* end() const;
};
static_assert( std::ranges::common_range<Range2>);
static_assert(!std::ranges::common_range<Range2 const>);

// Test ADL-proofing.
struct Incomplete;
template<class T> struct Holder { T t; };

static_assert(!std::ranges::common_range<Holder<Incomplete>*>);
static_assert(!std::ranges::common_range<Holder<Incomplete>*&>);
static_assert(!std::ranges::common_range<Holder<Incomplete>*&&>);
static_assert(!std::ranges::common_range<Holder<Incomplete>* const>);
static_assert(!std::ranges::common_range<Holder<Incomplete>* const&>);
static_assert(!std::ranges::common_range<Holder<Incomplete>* const&&>);

static_assert( std::ranges::common_range<Holder<Incomplete>*[10]>);
static_assert( std::ranges::common_range<Holder<Incomplete>*(&)[10]>);
static_assert( std::ranges::common_range<Holder<Incomplete>*(&&)[10]>);
static_assert( std::ranges::common_range<Holder<Incomplete>* const[10]>);
static_assert( std::ranges::common_range<Holder<Incomplete>* const(&)[10]>);
static_assert( std::ranges::common_range<Holder<Incomplete>* const(&&)[10]>);
