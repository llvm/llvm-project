//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <ranges>

// template <input_range V, forward_range Pattern>
//  requires view<V> && input_range<range_reference_t<V>> && view<Pattern> &&
//           compatible-joinable-ranges<range_reference_t<V>, Pattern>
// class join_with_view;

#include <ranges>

#include <cstddef>
#include <vector>

#include "test_iterators.h"

template <class View, class Pattern>
concept CanFormJoinWithView = requires { typename std::ranges::join_with_view<View, Pattern>; };

// join_with_view is not valid when `V` is not an input_range
namespace test_when_view_is_not_an_input_range {
struct View : std::ranges::view_base {
  using It = cpp20_output_iterator<std::vector<int>*>;
  It begin();
  sentinel_wrapper<It> end();
};

struct Pattern : std::ranges::view_base {
  int* begin();
  int* end();
};

static_assert(std::ranges::range<View>);
static_assert(!std::ranges::input_range<View>);
static_assert(std::ranges::view<View>);
static_assert(std::ranges::forward_range<Pattern>);
static_assert(std::ranges::view<Pattern>);
static_assert(!CanFormJoinWithView<View, Pattern>);
} // namespace test_when_view_is_not_an_input_range

// join_with_view is not valid when `Pattern` is not a forward_range
namespace test_when_pattern_is_not_a_forward_range {
struct View : std::ranges::view_base {
  std::vector<float>* begin();
  std::vector<float>* end();
};

struct Pattern : std::ranges::view_base {
  using It = cpp20_input_iterator<float*>;
  It begin();
  sentinel_wrapper<It> end();
};

static_assert(std::ranges::input_range<View>);
static_assert(std::ranges::view<View>);
static_assert(!std::ranges::forward_range<Pattern>);
static_assert(std::ranges::view<Pattern>);
static_assert(!CanFormJoinWithView<View, Pattern>);
} // namespace test_when_pattern_is_not_a_forward_range

// join_with_view is not valid when `V` does not model std::ranges::view
namespace test_when_view_does_not_model_view {
struct View {
  std::vector<double>* begin();
  std::vector<double>* end();
};

struct Pattern : std::ranges::view_base {
  double* begin();
  double* end();
};

static_assert(std::ranges::input_range<View>);
static_assert(!std::ranges::view<View>);
static_assert(std::ranges::forward_range<Pattern>);
static_assert(std::ranges::view<Pattern>);
static_assert(!CanFormJoinWithView<View, Pattern>);
} // namespace test_when_view_does_not_model_view

// join_with_view is not valid when `range_reference_t` of `V` is not an input_range
namespace test_when_range_reference_t_of_view_is_not_an_input_range {
struct InnerRange {
  using It = cpp20_output_iterator<long*>;
  It begin();
  sentinel_wrapper<It> end();
};

struct View : std::ranges::view_base {
  InnerRange* begin();
  InnerRange* end();
};

struct Pattern : std::ranges::view_base {
  long* begin();
  long* end();
};

static_assert(std::ranges::range<InnerRange>);
static_assert(!std::ranges::input_range<InnerRange>);
static_assert(std::ranges::input_range<View>);
static_assert(std::ranges::view<View>);
static_assert(std::ranges::forward_range<Pattern>);
static_assert(std::ranges::view<Pattern>);
static_assert(!CanFormJoinWithView<View, Pattern>);
} // namespace test_when_range_reference_t_of_view_is_not_an_input_range

// join_with_view is not valid when `Pattern` does not model std::ranges::view
namespace test_when_pattern_does_not_model_view {
struct View : std::ranges::view_base {
  std::vector<short>* begin();
  std::vector<short>* end();
};

struct Pattern {
  short* begin();
  short* end();
};

static_assert(std::ranges::input_range<View>);
static_assert(std::ranges::view<View>);
static_assert(std::ranges::forward_range<Pattern>);
static_assert(!std::ranges::view<Pattern>);
static_assert(!CanFormJoinWithView<View, Pattern>);
} // namespace test_when_pattern_does_not_model_view

// join_with_view is not valid when `range_reference_t<View>` and pattern
// does not model together compatible-joinable-ranges
namespace test_when_used_ranges_are_not_compatible_joinable_ranges {
using std::ranges::range_reference_t;
using std::ranges::range_rvalue_reference_t;
using std::ranges::range_value_t;

template <class InnerRange>
struct View : std::ranges::view_base {
  InnerRange* begin();
  InnerRange* end();
};

namespace no_common_range_value_type {
struct InnerRange {
  struct It {
    using difference_type = ptrdiff_t;
    struct value_type {};

    struct reference {
      operator value_type();
      operator float();
    };

    It& operator++();
    void operator++(int);
    reference operator*() const;
  };

  It begin();
  sentinel_wrapper<It> end();
};

struct Pattern : std::ranges::view_base {
  const float* begin();
  const float* end();
};

static_assert(std::ranges::input_range<InnerRange>);
static_assert(std::ranges::forward_range<Pattern>);
static_assert(std::ranges::view<Pattern>);
static_assert(!std::common_with<range_value_t<InnerRange>, range_value_t<Pattern>>);
static_assert(std::common_reference_with<range_reference_t<InnerRange>, range_reference_t<Pattern>>);
static_assert(std::common_reference_with<range_rvalue_reference_t<InnerRange>, range_rvalue_reference_t<Pattern>>);
static_assert(!CanFormJoinWithView<View<InnerRange>, Pattern>);
} // namespace no_common_range_value_type

namespace no_common_range_reference_type {
struct ValueType {};

struct InnerRange {
  struct It {
    using difference_type = ptrdiff_t;
    using value_type      = ValueType;
    struct reference {
      operator value_type();
    };

    It& operator++();
    void operator++(int);
    reference operator*() const;
  };

  It begin();
  sentinel_wrapper<It> end();
};

struct Pattern : std::ranges::view_base {
  struct It {
    using difference_type = ptrdiff_t;
    using value_type      = ValueType;
    struct reference {
      operator value_type();
    };

    It& operator++();
    It operator++(int);
    reference operator*() const;
    bool operator==(const It&) const;
    friend value_type&& iter_move(const It&);
  };

  It begin();
  It end();
};

static_assert(std::ranges::input_range<InnerRange>);
static_assert(std::ranges::forward_range<Pattern>);
static_assert(std::ranges::view<Pattern>);
static_assert(std::common_with<range_value_t<InnerRange>, range_value_t<Pattern>>);
static_assert(!std::common_reference_with<range_reference_t<InnerRange>, range_reference_t<Pattern>>);
static_assert(std::common_reference_with<range_rvalue_reference_t<InnerRange>, range_rvalue_reference_t<Pattern>>);
static_assert(!CanFormJoinWithView<View<InnerRange>, Pattern>);
} // namespace no_common_range_reference_type

namespace no_common_range_rvalue_reference_type {
struct InnerRange {
  using It = cpp20_input_iterator<int*>;
  It begin();
  sentinel_wrapper<It> end();
};

struct Pattern : std::ranges::view_base {
  struct It {
    using difference_type = ptrdiff_t;
    struct value_type {
      operator int() const;
    };

    struct rvalue_reference {
      operator value_type();
    };

    It& operator++();
    It operator++(int);
    value_type& operator*() const;
    bool operator==(const It&) const;
    friend rvalue_reference iter_move(const It&);
  };

  It begin();
  It end();
};

static_assert(std::ranges::input_range<InnerRange>);
static_assert(std::ranges::forward_range<Pattern>);
static_assert(std::ranges::view<Pattern>);
static_assert(std::common_with<range_value_t<InnerRange>, range_value_t<Pattern>>);
static_assert(std::common_reference_with<range_reference_t<InnerRange>, range_reference_t<Pattern>>);
static_assert(!std::common_reference_with<range_rvalue_reference_t<InnerRange>, range_rvalue_reference_t<Pattern>>);
static_assert(!CanFormJoinWithView<View<InnerRange>, Pattern>);
} // namespace no_common_range_rvalue_reference_type
} // namespace test_when_used_ranges_are_not_compatible_joinable_ranges
