//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

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

  // Test `view | views::stride`
  {
    using View                               = std::ranges::stride_view<BasicTestView<cpp17_input_iterator<int*>>>;
    auto view                                = make_input_view(arr, arr + N);
    std::same_as<View> decltype(auto) result = view | std::views::stride(2);
    auto it                                  = result.begin();

    assert(*it == arr[0]);
    std::ranges::advance(it, 1);
    assert(*it == arr[2]);
  }

  // Test `adaptor | views::stride`
  auto twice = [](int i) { return i * 2; };
  {
    using View = std::ranges::stride_view<
        std::ranges::transform_view<BasicTestView<cpp17_input_iterator<int*>>, decltype(twice)>>;
    auto view          = make_input_view(arr, arr + N);
    const auto partial = std::views::transform(twice) | std::views::stride(2);

    std::same_as<View> decltype(auto) result = partial(view);
    auto it                                  = result.begin();

    assert(*it == twice(arr[0]));
    std::ranges::advance(it, 1);
    assert(*it == twice(arr[2]));
  }

  // Test `views::stride | adaptor`
  {
    using View = std::ranges::transform_view< std::ranges::stride_view<BasicTestView<cpp17_input_iterator<int*>>>,
                                              decltype(twice)>;
    auto view  = make_input_view(arr, arr + N);
    std::same_as<View> decltype(auto) result = std::views::stride(view, 2) | std::views::transform(twice);

    auto it = result.begin();

    assert(*it == twice(arr[0]));
    std::ranges::advance(it, 1);
    assert(*it == twice(arr[2]));
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
