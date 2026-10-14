//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// test if adjacent_view models input_range, forward_range, bidirectional_range,
//                            random_access_range, contiguous_range, common_range
//                            sized_range

#include <cassert>
#include <concepts>
#include <ranges>
#include <tuple>
#include <utility>

#include "../range_adaptor_types.h"

template <std::size_t N>
constexpr bool testConcept() {
  int buffer[3] = {1, 2, 3};
  {
    std::ranges::adjacent_view<ContiguousCommonView, N> v(ContiguousCommonView{buffer});
    using View = decltype(v);
    static_assert(std::ranges::random_access_range<View>);
    static_assert(!std::ranges::contiguous_range<View>);
    static_assert(std::ranges::common_range<View>);
    static_assert(std::ranges::sized_range<View>);
  }

  {
    std::ranges::adjacent_view<ContiguousNonCommonView, N> v{ContiguousNonCommonView{buffer}};
    using View = decltype(v);
    static_assert(std::ranges::random_access_range<View>);
    static_assert(!std::ranges::contiguous_range<View>);
    static_assert(!std::ranges::common_range<View>);
    static_assert(!std::ranges::sized_range<View>);
  }

  {
    std::ranges::adjacent_view<ContiguousNonCommonSized, N> v{ContiguousNonCommonSized{buffer}};
    using View = decltype(v);
    static_assert(std::ranges::random_access_range<View>);
    static_assert(!std::ranges::contiguous_range<View>);
    static_assert(!std::ranges::common_range<View>);
    static_assert(std::ranges::sized_range<View>);
  }

  {
    std::ranges::adjacent_view<SizedRandomAccessView, N> v{SizedRandomAccessView{buffer}};
    using View = decltype(v);
    static_assert(std::ranges::random_access_range<View>);
    static_assert(!std::ranges::contiguous_range<View>);
    static_assert(!std::ranges::common_range<View>);
    static_assert(std::ranges::sized_range<View>);
  }

  {
    std::ranges::adjacent_view<NonSizedRandomAccessView, N> v{NonSizedRandomAccessView{buffer}};
    using View = decltype(v);
    static_assert(std::ranges::random_access_range<View>);
    static_assert(!std::ranges::contiguous_range<View>);
    static_assert(!std::ranges::common_range<View>);
    static_assert(!std::ranges::sized_range<View>);
  }

  {
    std::ranges::adjacent_view<BidiCommonView, N> v{BidiCommonView{buffer}};
    using View = decltype(v);
    static_assert(std::ranges::bidirectional_range<View>);
    static_assert(!std::ranges::random_access_range<View>);
    static_assert(std::ranges::common_range<View>);
    static_assert(!std::ranges::sized_range<View>);
  }

  {
    std::ranges::adjacent_view<BidiNonCommonView, N> v{BidiNonCommonView{buffer}};
    using View = decltype(v);
    static_assert(std::ranges::forward_range<View>);
    static_assert(std::ranges::bidirectional_range<View>);
    static_assert(!std::ranges::common_range<View>);
    static_assert(!std::ranges::sized_range<View>);
  }

  {
    std::ranges::adjacent_view<ForwardSizedView, N> v{ForwardSizedView{buffer}};
    using View = decltype(v);
    static_assert(std::ranges::forward_range<View>);
    static_assert(!std::ranges::bidirectional_range<View>);
    static_assert(std::ranges::common_range<View>);
    static_assert(std::ranges::sized_range<View>);
  }

  {
    std::ranges::adjacent_view<ForwardSizedNonCommon, N> v{ForwardSizedNonCommon{buffer}};
    using View = decltype(v);
    static_assert(std::ranges::forward_range<View>);
    static_assert(!std::ranges::bidirectional_range<View>);
    static_assert(!std::ranges::common_range<View>);
    static_assert(std::ranges::sized_range<View>);
  }

  return true;
}

static_assert(testConcept<1>());
static_assert(testConcept<2>());
static_assert(testConcept<3>());
static_assert(testConcept<5>());

using OutputIter = cpp17_output_iterator<int*>;
static_assert(std::output_iterator<OutputIter, int>);

struct OutputView : std::ranges::view_base {
  OutputIter begin() const;
  sentinel_wrapper<OutputIter> end() const;
};
static_assert(std::ranges::output_range<OutputView, int>);
static_assert(!std::ranges::input_range<OutputView>);

template <class T, std::size_t N>
concept adjacent_viewable = requires { typename std::ranges::adjacent_view<T, N>; };

static_assert(adjacent_viewable<SimpleCommon, 2>);

// output_range is not supported
static_assert(!adjacent_viewable<OutputView, 2>);

// input only range is not supported
static_assert(!adjacent_viewable<InputCommonView, 1>);
static_assert(!adjacent_viewable<InputNonCommonView, 2>);
