//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// [iterator.concept.sizedsentinel], concept sized_sentinel_for
//
// template<class S, class I>
//   inline constexpr bool disable_sized_sentinel_for = false;
//
// template<class S, class I>
//   concept sized_sentinel_for = see below;

#include <iterator>

#include <array>
#include <concepts>
#include <deque>
#include <string>
#include <string_view>
#include <vector>

#include "test_iterators.h"
#include "test_macros.h"

static_assert(std::sized_sentinel_for<random_access_iterator<int*>, random_access_iterator<int*> >);
static_assert(!std::sized_sentinel_for<bidirectional_iterator<int*>, bidirectional_iterator<int*> >);

struct int_sized_sentinel {
  friend bool operator==(int_sized_sentinel, int*);
  friend std::ptrdiff_t operator-(int_sized_sentinel, int*);
  friend std::ptrdiff_t operator-(int*, int_sized_sentinel);
};
static_assert(std::sized_sentinel_for<int_sized_sentinel, int*>);
// int_sized_sentinel is not an iterator.
static_assert(!std::sized_sentinel_for<int*, int_sized_sentinel>);

struct no_default_ctor {
  no_default_ctor() = delete;
  bool operator==(std::input_or_output_iterator auto) const;
};
static_assert(!std::sized_sentinel_for<no_default_ctor, int*>);

struct not_copyable {
  not_copyable() = default;
  not_copyable(not_copyable const&) = delete;
  bool operator==(std::input_or_output_iterator auto) const;
};
static_assert(!std::sized_sentinel_for<not_copyable, int*>);

struct double_sized_sentinel {
  friend bool operator==(double_sized_sentinel, double*);
  friend int operator-(double_sized_sentinel, double*);
  friend int operator-(double*, double_sized_sentinel);
};
template <>
inline constexpr bool std::disable_sized_sentinel_for<double_sized_sentinel, double*> = true;

static_assert(!std::sized_sentinel_for<double_sized_sentinel, double*>);

struct only_one_sub_op {
  friend bool operator==(only_one_sub_op, std::input_or_output_iterator auto);
  friend std::ptrdiff_t operator-(only_one_sub_op, std::input_or_output_iterator auto);
};
static_assert(!std::sized_sentinel_for<only_one_sub_op, int*>);

struct wrong_return_type {
  friend bool operator==(wrong_return_type, std::input_or_output_iterator auto);
  friend std::ptrdiff_t operator-(wrong_return_type, std::input_or_output_iterator auto);
  friend void operator-(std::input_or_output_iterator auto, wrong_return_type);
};
static_assert(!std::sized_sentinel_for<wrong_return_type, int*>);

// Standard types
static_assert(std::sized_sentinel_for<int*, int*>);
static_assert(std::sized_sentinel_for<const int*, int*>);
static_assert(std::sized_sentinel_for<const int*, const int*>);
static_assert(std::sized_sentinel_for<int*, const int*>);
