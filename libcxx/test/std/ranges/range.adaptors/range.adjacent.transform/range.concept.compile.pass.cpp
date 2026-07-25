//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// test if adjacent_transform_view models input_range, forward_range, bidirectional_range,
//  random_access_range, contiguous_range, common_range, sized_range

#include <cassert>
#include <concepts>
#include <ranges>
#include <tuple>
#include <utility>

#include "helpers.h"
#include "../range_adaptor_types.h"

template <std::size_t N, class Fn>
constexpr void testConcept() {
  int buffer[3] = {1, 2, 3};
  {
    std::ranges::adjacent_transform_view<ContiguousCommonView, Fn, N> v(ContiguousCommonView{buffer}, Fn{});
    using View = decltype(v);
    static_assert(std::ranges::random_access_range<View>);
    static_assert(!std::ranges::contiguous_range<View>);
    static_assert(std::ranges::common_range<View>);
    static_assert(std::ranges::sized_range<View>);
  }

  {
    std::ranges::adjacent_transform_view<ContiguousNonCommonView, Fn, N> v{ContiguousNonCommonView{buffer}, Fn{}};
    using View = decltype(v);
    static_assert(std::ranges::random_access_range<View>);
    static_assert(!std::ranges::contiguous_range<View>);
    static_assert(!std::ranges::common_range<View>);
    static_assert(!std::ranges::sized_range<View>);
  }

  {
    std::ranges::adjacent_transform_view<ContiguousNonCommonSized, Fn, N> v{ContiguousNonCommonSized{buffer}, Fn{}};
    using View = decltype(v);
    static_assert(std::ranges::random_access_range<View>);
    static_assert(!std::ranges::contiguous_range<View>);
    static_assert(!std::ranges::common_range<View>);
    static_assert(std::ranges::sized_range<View>);
  }

  {
    std::ranges::adjacent_transform_view<SizedRandomAccessView, Fn, N> v{SizedRandomAccessView{buffer}, Fn{}};
    using View = decltype(v);
    static_assert(std::ranges::random_access_range<View>);
    static_assert(!std::ranges::contiguous_range<View>);
    static_assert(!std::ranges::common_range<View>);
    static_assert(std::ranges::sized_range<View>);
  }

  {
    std::ranges::adjacent_transform_view<NonSizedRandomAccessView, Fn, N> v{NonSizedRandomAccessView{buffer}, Fn{}};
    using View = decltype(v);
    static_assert(std::ranges::random_access_range<View>);
    static_assert(!std::ranges::contiguous_range<View>);
    static_assert(!std::ranges::common_range<View>);
    static_assert(!std::ranges::sized_range<View>);
  }

  {
    std::ranges::adjacent_transform_view<BidiCommonView, Fn, N> v{BidiCommonView{buffer}, Fn{}};
    using View = decltype(v);
    static_assert(std::ranges::bidirectional_range<View>);
    static_assert(!std::ranges::random_access_range<View>);
    static_assert(std::ranges::common_range<View>);
    static_assert(!std::ranges::sized_range<View>);
  }

  {
    std::ranges::adjacent_transform_view<BidiNonCommonView, Fn, N> v{BidiNonCommonView{buffer}, Fn{}};
    using View = decltype(v);
    static_assert(std::ranges::forward_range<View>);
    static_assert(std::ranges::bidirectional_range<View>);
    static_assert(!std::ranges::common_range<View>);
    static_assert(!std::ranges::sized_range<View>);
  }

  {
    std::ranges::adjacent_transform_view<ForwardSizedView, Fn, N> v{ForwardSizedView{buffer}, Fn{}};
    using View = decltype(v);
    static_assert(std::ranges::forward_range<View>);
    static_assert(!std::ranges::bidirectional_range<View>);
    static_assert(std::ranges::common_range<View>);
    static_assert(std::ranges::sized_range<View>);
  }

  {
    std::ranges::adjacent_transform_view<ForwardSizedNonCommon, Fn, N> v{ForwardSizedNonCommon{buffer}, Fn{}};
    using View = decltype(v);
    static_assert(std::ranges::forward_range<View>);
    static_assert(!std::ranges::bidirectional_range<View>);
    static_assert(!std::ranges::common_range<View>);
    static_assert(std::ranges::sized_range<View>);
  }
}

template <std::size_t N>
constexpr bool testConcept() {
  testConcept<N, MakeTuple>();
  testConcept<N, Tie>();
  testConcept<N, GetFirst>();
  testConcept<N, Multiply>();

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

template <class T, class Fn, std::size_t N>
concept adjacent_transform_viewable = requires { typename std::ranges::adjacent_transform_view<T, Fn, N>; };

static_assert(adjacent_transform_viewable<SimpleCommon, MakeTuple, 2>);
static_assert(adjacent_transform_viewable<SimpleCommon, Tie, 2>);
static_assert(adjacent_transform_viewable<SimpleCommon, GetFirst, 2>);

// output_range is not supported
static_assert(!adjacent_transform_viewable<OutputView, MakeTuple, 2>);

// input only range is not supported
static_assert(!adjacent_transform_viewable<InputCommonView, MakeTuple, 1>);
static_assert(!adjacent_transform_viewable<InputNonCommonView, MakeTuple, 2>);

// Fn is not callble with the correct types
struct FnNotCallable {
  void operator()() const {}
};
static_assert(!adjacent_transform_viewable<SimpleCommon, FnNotCallable, 2>);

// function that returns void is not supported
struct FnReturnsVoid {
  template <class... Args>
  void operator()(Args&&...) const {}
};
static_assert(!adjacent_transform_viewable<SimpleCommon, FnReturnsVoid, 2>);
