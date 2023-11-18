//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// ranges

// std::views::stride_view

#include "__ranges/stride_view.h"
#include "test.h"
#include "test_iterators.h"
#include <concepts>
#include <iterator>
#include <ranges>
#include <utility>

template <class View, class T>
concept CanBePiped = requires(View&& view, T&& t) {
  { std::forward<View>(view) | std::forward<T>(t) };
};

constexpr bool test() {
  constexpr int array_n = 3;
  int arr[array_n]      = {1, 2, 3};

  // Test that `std::views::stride` is a range adaptor.
  { // Check various forms of
    // view | stride
    {
      {
        using View = InputView<bidirectional_iterator<int*>>;
        View view(bidirectional_iterator<int*>(arr), bidirectional_iterator<int*>(arr + array_n));
        std::same_as<std::ranges::stride_view<View>> decltype(auto) strided = view | std::views::stride(1);
        auto strided_iter                                                   = strided.begin();

        // Check that the begin() iter views arr[0]
        assert(*strided_iter == arr[0]);

        // Check that the strided_iter, after advancing it 2 * 1 steps, views arr[2].
        std::ranges::advance(strided_iter, 2);
        assert(*strided_iter == arr[2]);
      }
      {
        using View = InputView<bidirectional_iterator<int*>>;
        View view(bidirectional_iterator<int*>(arr), bidirectional_iterator<int*>(arr + array_n));
        std::same_as<std::ranges::stride_view<View>> decltype(auto) strided = view | std::views::stride(2);
        auto strided_iter                                                   = strided.begin();

        assert(*strided_iter == arr[0]);

        // Same test as above, just advance one time with a bigger step (1 * 2 steps).
        std::ranges::advance(strided_iter, 1);
        assert(*strided_iter == arr[2]);
      }
    }
  }

  // Check various forms of
  // adaptor | stride
  {
    // Parallels the two tests from above.
    constexpr const auto identity_lambda = [](const int i) { return i * 2; };
    {
      using View = InputView<bidirectional_iterator<int*>>;
      View view(bidirectional_iterator<int*>(arr), bidirectional_iterator<int*>(arr + array_n));
      const auto transform_stride_partial = std::views::transform(identity_lambda) | std::views::stride(1);

      auto transform_stride_applied      = transform_stride_partial(view);
      auto transform_stride_applied_iter = transform_stride_applied.begin();
      assert(*transform_stride_applied_iter == std::invoke(identity_lambda, arr[0]));
      std::ranges::advance(transform_stride_applied_iter, 2);
      assert(*transform_stride_applied_iter == std::invoke(identity_lambda, arr[2]));
    }

    {
      using View = InputView<bidirectional_iterator<int*>>;
      View view(bidirectional_iterator<int*>(arr), bidirectional_iterator<int*>(arr + array_n));
      const auto transform_stride_partial = std::views::transform(identity_lambda) | std::views::stride(2);

      const auto transform_stride_applied = transform_stride_partial(view);
      auto transform_stride_applied_iter  = transform_stride_applied.begin();
      assert(*transform_stride_applied_iter == std::invoke(identity_lambda, arr[0]));
      std::ranges::advance(transform_stride_applied_iter, 1);
      assert(*transform_stride_applied_iter == std::invoke(identity_lambda, arr[2]));
    }
  }

  {
    using ForwardStrideView      = std::ranges::stride_view<InputView<forward_iterator<int*>>>;
    using BidirStrideView        = std::ranges::stride_view<InputView<bidirectional_iterator<int*>>>;
    using RandomAccessStrideView = std::ranges::stride_view<InputView<random_access_iterator<int*>>>;

    static_assert(std::ranges::forward_range<ForwardStrideView>);
    static_assert(std::ranges::bidirectional_range<BidirStrideView>);
    static_assert(std::ranges::random_access_range<RandomAccessStrideView>);
    // TODO: check sized_range
  }

  // Check SFINAE friendliness
  {
    using View = InputView<bidirectional_iterator<int*>>;
    struct NotAViewableRange {};
    struct NotARange {};
    // Not invocable because there is no parameter.
    static_assert(!std::is_invocable_v<decltype(std::views::stride)>);
    // Not invocable because NotAViewableRange is, well, not a viewable range.
    static_assert(!std::is_invocable_v<decltype(std::views::reverse), NotAViewableRange>);
    // Is invocable because BidirView is a viewable range.
    static_assert(std::is_invocable_v<decltype(std::views::reverse), View>);

    // Make sure that pipe operations work!
    static_assert(CanBePiped<View, decltype(std::views::stride(std::ranges::range_difference_t<View>{}))>);
    static_assert(CanBePiped<View&, decltype(std::views::stride(std::ranges::range_difference_t<View>{}))>);
    static_assert(!CanBePiped<NotARange, decltype(std::views::stride(std::ranges::range_difference_t<View>{}))>);
  }
  // A final sanity check.
  { static_assert(std::same_as<decltype(std::views::stride), decltype(std::ranges::views::stride)>); }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
