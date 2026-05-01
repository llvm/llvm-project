//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// Test which range concepts cartesian_product_view models for various input
// range categories. The relevant rules are:
//
//   * random_access     iff first is random_access AND every other is random_access + sized
//   * bidirectional     iff first is bidi          AND every other is bidi + (common or random+sized)
//   * forward           iff first is forward
//   * input             always (when valid)
//   * sized             iff every range is sized
//   * common            iff first is common or (sized + random_access)

#include <ranges>

#include "test_iterators.h"

#include "../range_adaptor_types.h"

void testConceptPair() {
  int b1[2] = {1, 2};
  int b2[3] = {1, 2, 3};

  {
    std::ranges::cartesian_product_view v{ContiguousCommonView{b1}, ContiguousCommonView{b2}};
    using View = decltype(v);
    static_assert(std::ranges::random_access_range<View>);
    static_assert(!std::ranges::contiguous_range<View>);
    static_assert(std::ranges::common_range<View>);
    static_assert(std::ranges::sized_range<View>);
  }

  {
    std::ranges::cartesian_product_view v{SizedRandomAccessView{b1}, SizedRandomAccessView{b2}};
    using View = decltype(v);
    static_assert(std::ranges::random_access_range<View>);
    static_assert(std::ranges::common_range<View>);
    static_assert(std::ranges::sized_range<View>);
  }

  { // first is sized+random_access, so satisfies the common-arg condition despite not being common_range
    std::ranges::cartesian_product_view v{ContiguousNonCommonSized{b1}, ContiguousCommonView{b2}};
    using View = decltype(v);
    static_assert(std::ranges::random_access_range<View>);
    static_assert(std::ranges::common_range<View>);
    static_assert(std::ranges::sized_range<View>);
  }

  { // first is neither common nor sized+random_access, so not a common-arg -> not common, not sized
    std::ranges::cartesian_product_view v{ContiguousNonCommonView{b1}, ContiguousCommonView{b2}};
    using View = decltype(v);
    static_assert(!std::ranges::common_range<View>);
    static_assert(!std::ranges::sized_range<View>);
  }

  {
    std::ranges::cartesian_product_view v{BidiCommonView{b1}, SizedRandomAccessView{b2}};
    using View = decltype(v);
    static_assert(std::ranges::bidirectional_range<View>);
    static_assert(!std::ranges::random_access_range<View>);
    static_assert(std::ranges::common_range<View>);
    static_assert(!std::ranges::sized_range<View>);
  }

  {
    std::ranges::cartesian_product_view v{BidiCommonView{b1}, BidiCommonView{b2}};
    using View = decltype(v);
    static_assert(std::ranges::bidirectional_range<View>);
    static_assert(!std::ranges::random_access_range<View>);
    static_assert(std::ranges::common_range<View>);
    static_assert(!std::ranges::sized_range<View>);
  }

  {
    std::ranges::cartesian_product_view v{BidiCommonView{b1}, ForwardSizedView{b2}};
    using View = decltype(v);
    static_assert(std::ranges::forward_range<View>);
    static_assert(!std::ranges::bidirectional_range<View>);
    static_assert(std::ranges::common_range<View>);
    static_assert(!std::ranges::sized_range<View>);
  }

  {
    std::ranges::cartesian_product_view v{BidiNonCommonView{b1}, ForwardSizedView{b2}};
    using View = decltype(v);
    static_assert(std::ranges::forward_range<View>);
    static_assert(!std::ranges::common_range<View>);
    static_assert(!std::ranges::sized_range<View>);
  }

  {
    std::ranges::cartesian_product_view v{ForwardSizedView{b1}, ForwardSizedView{b2}};
    using View = decltype(v);
    static_assert(std::ranges::forward_range<View>);
    static_assert(!std::ranges::bidirectional_range<View>);
    static_assert(std::ranges::common_range<View>);
    static_assert(std::ranges::sized_range<View>);
  }

  {
    std::ranges::cartesian_product_view v{ForwardSizedNonCommon{b1}, ForwardSizedView{b2}};
    using View = decltype(v);
    static_assert(std::ranges::forward_range<View>);
    static_assert(!std::ranges::common_range<View>);
    static_assert(std::ranges::sized_range<View>);
  }

  {
    std::ranges::cartesian_product_view v{InputCommonView{b1}, ForwardSizedView{b2}};
    using View = decltype(v);
    static_assert(std::ranges::input_range<View>);
    static_assert(!std::ranges::forward_range<View>);
    static_assert(std::ranges::common_range<View>);
    static_assert(!std::ranges::sized_range<View>);
  }

  {
    std::ranges::cartesian_product_view v{InputNonCommonView{b1}, ForwardSizedView{b2}};
    using View = decltype(v);
    static_assert(std::ranges::input_range<View>);
    static_assert(!std::ranges::forward_range<View>);
    static_assert(!std::ranges::common_range<View>);
    static_assert(!std::ranges::sized_range<View>);
  }
}

void testConceptTuple() {
  int b1[2] = {1, 2};
  int b2[3] = {1, 2, 3};
  int b3[4] = {1, 2, 3, 4};

  {
    std::ranges::cartesian_product_view v{ContiguousCommonView{b1}, ContiguousCommonView{b2}, ContiguousCommonView{b3}};
    using View = decltype(v);
    static_assert(std::ranges::random_access_range<View>);
    static_assert(std::ranges::common_range<View>);
    static_assert(std::ranges::sized_range<View>);
  }

  {
    std::ranges::cartesian_product_view v{BidiCommonView{b1}, BidiCommonView{b2}, BidiCommonView{b3}};
    using View = decltype(v);
    static_assert(std::ranges::bidirectional_range<View>);
    static_assert(!std::ranges::random_access_range<View>);
    static_assert(std::ranges::common_range<View>);
    static_assert(!std::ranges::sized_range<View>);
  }

  {
    std::ranges::cartesian_product_view v{BidiCommonView{b1}, ForwardSizedView{b2}, ForwardSizedView{b3}};
    using View = decltype(v);
    static_assert(std::ranges::forward_range<View>);
    static_assert(!std::ranges::bidirectional_range<View>);
    static_assert(std::ranges::common_range<View>);
    static_assert(!std::ranges::sized_range<View>);
  }

  {
    std::ranges::cartesian_product_view v{InputCommonView{b1}, ForwardSizedView{b2}, ForwardSizedView{b3}};
    using View = decltype(v);
    static_assert(std::ranges::input_range<View>);
    static_assert(!std::ranges::forward_range<View>);
    static_assert(std::ranges::common_range<View>);
    static_assert(!std::ranges::sized_range<View>);
  }
}

// cartesian_product_view requires the first range to be input_range and all others to be forward_range.
struct OutputView : std::ranges::view_base {
  cpp17_output_iterator<int*> begin() const;
  sentinel_wrapper<cpp17_output_iterator<int*>> end() const;
};
static_assert(std::ranges::output_range<OutputView, int>);
static_assert(!std::ranges::input_range<OutputView>);

template <class... Ts>
concept cartesian_constructible = requires { typename std::ranges::cartesian_product_view<Ts...>; };

static_assert(!cartesian_constructible<OutputView>);
static_assert(!cartesian_constructible<SimpleCommon, OutputView>);
static_assert(!cartesian_constructible<SimpleCommon, InputCommonView>);
static_assert(cartesian_constructible<SimpleCommon>);
static_assert(cartesian_constructible<InputCommonView, ForwardSizedView>);
