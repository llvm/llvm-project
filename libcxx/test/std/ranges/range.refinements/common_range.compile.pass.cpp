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
// concept common_range;

#include <ranges>

#include "test_iterators.h"
#include "test_range.h"

namespace stdr = std::ranges;

static_assert(!stdr::common_range<test_range<cpp17_input_iterator> >);
static_assert(!stdr::common_range<test_range<cpp17_input_iterator> const>);

static_assert(!stdr::common_range<test_non_const_range<cpp17_input_iterator> >);
static_assert(!stdr::common_range<test_non_const_range<cpp17_input_iterator> const>);

static_assert(stdr::common_range<test_common_range<cpp17_input_iterator> >);
static_assert(stdr::common_range<test_common_range<cpp17_input_iterator> const>);

static_assert(stdr::common_range<test_non_const_common_range<cpp17_input_iterator> >);
static_assert(!stdr::common_range<test_non_const_common_range<cpp17_input_iterator> const>);

struct subtly_not_common {
  int* begin() const;
  int const* end() const;
};
static_assert(stdr::range<subtly_not_common> && !stdr::common_range<subtly_not_common>);
static_assert(stdr::range<subtly_not_common const> && !stdr::common_range<subtly_not_common const>);

struct common_range_non_const_only {
  int* begin() const;
  int* end();
  int const* end() const;
};
static_assert(stdr::range<common_range_non_const_only>&& stdr::common_range<common_range_non_const_only>);
static_assert(stdr::range<common_range_non_const_only const> && !stdr::common_range<common_range_non_const_only const>);

struct common_range_const_only {
  int* begin();
  int const* begin() const;
  int const* end() const;
};
static_assert(stdr::range<common_range_const_only> && !stdr::common_range<common_range_const_only>);
static_assert(stdr::range<common_range_const_only const>&& stdr::common_range<common_range_const_only const>);
