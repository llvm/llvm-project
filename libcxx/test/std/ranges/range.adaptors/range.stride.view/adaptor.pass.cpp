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

#include "test.h"
#include <iterator>
#include <ranges>
#include <utility>

template <class View, class T>
concept CanBePiped = requires(View&& view, T&& t) {
  { std::forward<View>(view) | std::forward<T>(t) };
};

constexpr bool test() {
  int arr[] = {1, 2, 3};

  // Simple use cases.
  {
    {
      BidirView view(arr, arr + 3);
      std::ranges::stride_view<BidirView> strided(view, 1);
      auto strided_iter = strided.begin();

      assert(*strided_iter == arr[0]);

      std::ranges::advance(strided_iter, 2);
      assert(*strided_iter == arr[2]);
    }
    {
      BidirView view(arr, arr + 3);
      std::ranges::stride_view<BidirView> strided(view, 2);
      auto strided_iter = strided.begin();

      assert(*strided_iter == arr[0]);

      std::ranges::advance(strided_iter, 1);
      assert(*strided_iter == arr[2]);
    }
  }

#if 0
  // views::reverse(x) is equivalent to subrange{end, begin, size} if x is a
  // sized subrange over reverse iterators
  {
    using It = bidirectional_iterator<int*>;
    using Subrange = std::ranges::subrange<It, It, std::ranges::subrange_kind::sized>;

    using ReverseIt = std::reverse_iterator<It>;
    using ReverseSubrange = std::ranges::subrange<ReverseIt, ReverseIt, std::ranges::subrange_kind::sized>;

    {
      BidirRange view(buf, buf + 3);
      ReverseSubrange subrange(ReverseIt(std::ranges::end(view)), ReverseIt(std::ranges::begin(view)), /* size */3);
      std::same_as<Subrange> auto result = std::views::reverse(subrange);
      assert(base(result.begin()) == buf);
      assert(base(result.end()) == buf + 3);
    }
    {
      // std::move into views::reverse
      BidirRange view(buf, buf + 3);
      ReverseSubrange subrange(ReverseIt(std::ranges::end(view)), ReverseIt(std::ranges::begin(view)), /* size */3);
      std::same_as<Subrange> auto result = std::views::reverse(std::move(subrange));
      assert(base(result.begin()) == buf);
      assert(base(result.end()) == buf + 3);
    }
    {
      // with a const subrange
      BidirRange view(buf, buf + 3);
      ReverseSubrange const subrange(ReverseIt(std::ranges::end(view)), ReverseIt(std::ranges::begin(view)), /* size */3);
      std::same_as<Subrange> auto result = std::views::reverse(subrange);
      assert(base(result.begin()) == buf);
      assert(base(result.end()) == buf + 3);
    }
  }

  // views::reverse(x) is equivalent to subrange{end, begin} if x is an
  // unsized subrange over reverse iterators
  {
    using It = bidirectional_iterator<int*>;
    using Subrange = std::ranges::subrange<It, It, std::ranges::subrange_kind::unsized>;

    using ReverseIt = std::reverse_iterator<It>;
    using ReverseSubrange = std::ranges::subrange<ReverseIt, ReverseIt, std::ranges::subrange_kind::unsized>;

    {
      BidirRange view(buf, buf + 3);
      ReverseSubrange subrange(ReverseIt(std::ranges::end(view)), ReverseIt(std::ranges::begin(view)));
      std::same_as<Subrange> auto result = std::views::reverse(subrange);
      assert(base(result.begin()) == buf);
      assert(base(result.end()) == buf + 3);
    }
    {
      // std::move into views::reverse
      BidirRange view(buf, buf + 3);
      ReverseSubrange subrange(ReverseIt(std::ranges::end(view)), ReverseIt(std::ranges::begin(view)));
      std::same_as<Subrange> auto result = std::views::reverse(std::move(subrange));
      assert(base(result.begin()) == buf);
      assert(base(result.end()) == buf + 3);
    }
    {
      // with a const subrange
      BidirRange view(buf, buf + 3);
      ReverseSubrange const subrange(ReverseIt(std::ranges::end(view)), ReverseIt(std::ranges::begin(view)));
      std::same_as<Subrange> auto result = std::views::reverse(subrange);
      assert(base(result.begin()) == buf);
      assert(base(result.end()) == buf + 3);
    }
  }

  // Otherwise, views::reverse(x) is equivalent to ranges::reverse_view{x}
  {
    BidirRange view(buf, buf + 3);
    std::same_as<std::ranges::reverse_view<BidirRange>> auto result = std::views::reverse(view);
    assert(base(result.begin().base()) == buf + 3);
    assert(base(result.end().base()) == buf);
  }

  // Test that std::views::reverse is a range adaptor
  {
    // Test `v | views::reverse`
    {
      BidirRange view(buf, buf + 3);
      std::same_as<std::ranges::reverse_view<BidirRange>> auto result = view | std::views::reverse;
      assert(base(result.begin().base()) == buf + 3);
      assert(base(result.end().base()) == buf);
    }

    // Test `adaptor | views::reverse`
    {
      BidirRange view(buf, buf + 3);
      auto f = [](int i) { return i; };
      auto const partial = std::views::transform(f) | std::views::reverse;
      using Result = std::ranges::reverse_view<std::ranges::transform_view<BidirRange, decltype(f)>>;
      std::same_as<Result> auto result = partial(view);
      assert(base(result.begin().base().base()) == buf + 3);
      assert(base(result.end().base().base()) == buf);
    }

    // Test `views::reverse | adaptor`
    {
      BidirRange view(buf, buf + 3);
      auto f = [](int i) { return i; };
      auto const partial = std::views::reverse | std::views::transform(f);
      using Result = std::ranges::transform_view<std::ranges::reverse_view<BidirRange>, decltype(f)>;
      std::same_as<Result> auto result = partial(view);
      assert(base(result.begin().base().base()) == buf + 3);
      assert(base(result.end().base().base()) == buf);
    }
  }
#endif // big block

  // From:
  // Test that std::views::reverse is a range adaptor
  // Check SFINAE friendliness
  {
    struct NotAViewableRange {};
    struct NotABidirRange {};
    // Not invocable because there is no parameter.
    static_assert(!std::is_invocable_v<decltype(std::views::stride)>);
    // Not invocable because NotAViewableRange is, well, not a viewable range.
    static_assert(!std::is_invocable_v<decltype(std::views::reverse), NotAViewableRange>);
    // Is invocable because BidirView is a viewable range.
    static_assert(std::is_invocable_v<decltype(std::views::reverse), BidirView>);

    // Make sure that pipe operations work!
    static_assert(CanBePiped<BidirView, decltype(std::views::stride(std::ranges::range_difference_t<BidirView>{}))>);
    static_assert(CanBePiped<BidirView&, decltype(std::views::stride(std::ranges::range_difference_t<BidirView>{}))>);
    static_assert(
        !CanBePiped<NotABidirRange, decltype(std::views::stride(std::ranges::range_difference_t<BidirView>{}))>);
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
