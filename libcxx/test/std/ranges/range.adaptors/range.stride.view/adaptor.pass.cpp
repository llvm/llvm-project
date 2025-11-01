//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// std::views::stride_view

#include <ranges>

#include "test_iterators.h"
#include "types.h"

constexpr BasicTestView<cpp17_input_iterator<int*>> make_input_view(int* begin, int* end) {
  return BasicTestView<cpp17_input_iterator<int*>>(cpp17_input_iterator<int*>(begin), cpp17_input_iterator<int*>(end));
}

using ForwardStrideView      = std::ranges::stride_view<BasicTestView<forward_iterator<int*>>>;
using BidirStrideView        = std::ranges::stride_view<BasicTestView<bidirectional_iterator<int*>>>;
using RandomAccessStrideView = std::ranges::stride_view<BasicTestView<random_access_iterator<int*>>>;

using SizedForwardStrideView =
    std::ranges::stride_view<BasicTestView<random_access_iterator<int*>, random_access_iterator<int*>>>;
using SizedInputStrideView = std::ranges::stride_view<BasicTestView<SizedInputIter, SizedInputIter>>;

static_assert(std::ranges::forward_range<ForwardStrideView>);
static_assert(std::ranges::bidirectional_range<BidirStrideView>);
static_assert(std::ranges::random_access_range<RandomAccessStrideView>);
static_assert(std::ranges::forward_range<SizedForwardStrideView>);
static_assert(std::sized_sentinel_for<std::ranges::iterator_t<SizedForwardStrideView>,
                                      std::ranges::iterator_t<SizedForwardStrideView>>);
static_assert(std::sized_sentinel_for<std::ranges::iterator_t<SizedInputStrideView>,
                                      std::ranges::iterator_t<SizedInputStrideView>>);

constexpr bool test() {
  constexpr int N = 3;
  int arr[N]      = {1, 2, 3};

  // Test that `std::views::stride` is a range adaptor.
  // Check various forms of

  // view | stride
  // two tests: first with stride of 1; second with stride of 2.
  {
    using View                                                          = BasicTestView<cpp17_input_iterator<int*>>;
    auto view                                                           = make_input_view(arr, arr + N);
    std::same_as<std::ranges::stride_view<View>> decltype(auto) strided = view | std::views::stride(1);
    auto strided_iter                                                   = strided.begin();

    // Check that the begin() iter views arr[0]
    assert(*strided_iter == arr[0]);

    // Check that the strided_iter, after advancing it 2 * 1 steps, views arr[2].
    std::ranges::advance(strided_iter, 2);
    assert(*strided_iter == arr[2]);
  }
  {
    using View                                                          = BasicTestView<cpp17_input_iterator<int*>>;
    auto view                                                           = make_input_view(arr, arr + N);
    std::same_as<std::ranges::stride_view<View>> decltype(auto) strided = view | std::views::stride(2);
    auto strided_iter                                                   = strided.begin();

    assert(*strided_iter == arr[0]);
    std::ranges::advance(strided_iter, 1);
    assert(*strided_iter == arr[2]);
  }

  // adaptor | stride
  // two tests: first with stride of 1; second with stride of 2.
  const auto i2 = [](int i) { return i * 2; };
  {
    auto view                           = make_input_view(arr, arr + N);
    const auto transform_stride_partial = std::views::transform(i2) | std::views::stride(1);

    auto transform_stride_applied      = transform_stride_partial(view);
    auto transform_stride_applied_iter = transform_stride_applied.begin();

    assert(*transform_stride_applied_iter == i2(arr[0]));
    std::ranges::advance(transform_stride_applied_iter, 2);
    assert(*transform_stride_applied_iter == i2(arr[2]));
  }

  {
    auto view                           = make_input_view(arr, arr + N);
    const auto transform_stride_partial = std::views::transform(i2) | std::views::stride(2);

    const auto transform_stride_applied = transform_stride_partial(view);
    auto transform_stride_applied_iter  = transform_stride_applied.begin();

    assert(*transform_stride_applied_iter == i2(arr[0]));
    std::ranges::advance(transform_stride_applied_iter, 1);
    assert(*transform_stride_applied_iter == i2(arr[2]));
  }

  // stride | adaptor
  // two tests: first with stride of 1; second with stride of 2.
  {
    auto view                   = make_input_view(arr, arr + N);
    const auto stride_transform = std::views::stride(view, 1) | std::views::transform(i2);

    auto stride_transform_iter = stride_transform.begin();

    assert(*stride_transform_iter == i2(arr[0]));
    std::ranges::advance(stride_transform_iter, 2);
    assert(*stride_transform_iter == i2(arr[2]));
  }
  {
    auto view                   = make_input_view(arr, arr + N);
    const auto stride_transform = std::views::stride(view, 2) | std::views::transform(i2);

    auto stride_transform_iter = stride_transform.begin();

    assert(*stride_transform_iter == i2(arr[0]));
    std::ranges::advance(stride_transform_iter, 1);
    assert(*stride_transform_iter == i2(arr[2]));
  }

  // Check SFINAE friendliness
  {
    struct NotAViewableRange {};
    using View = BasicTestView<bidirectional_iterator<int*>>;

    static_assert(!std::is_invocable_v<decltype(std::views::stride)>);
    static_assert(!std::is_invocable_v<decltype(std::views::stride), NotAViewableRange, int>);

    static_assert(CanBePiped<View, decltype(std::views::stride(5))>);
    static_assert(CanBePiped<View&, decltype(std::views::stride(5))>);
    static_assert(!CanBePiped<NotAViewableRange, decltype(std::views::stride(5))>);
    static_assert(!CanBePiped<View&, decltype(std::views::stride(NotAViewableRange{}))>);
  }

  // A final sanity check.
  {
    static_assert(std::same_as<decltype(std::views::stride), decltype(std::ranges::views::stride)>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
