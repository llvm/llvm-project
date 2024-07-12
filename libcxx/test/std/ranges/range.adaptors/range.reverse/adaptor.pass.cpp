//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// std::views::reverse

#include <ranges>

#include <cassert>
#include <concepts>
#include <iterator>
#include <utility>

#include "test_range.h"
#include "types.h"

constexpr bool test() {
  int buf[] = {1, 2, 3};

  // views::reverse(x) is equivalent to x.base() if x is a reverse_view
  {
    {
      BidirRange view(buf, buf + 3);
      std::ranges::reverse_view<BidirRange> reversed(view);
      std::same_as<BidirRange> auto result = std::views::reverse(reversed);
      assert(result.begin_ == buf);
      assert(result.end_ == buf + 3);
    }
    {
      // Common use case is worth testing
      BidirRange view(buf, buf + 3);
      std::same_as<BidirRange> auto result = std::views::reverse(std::views::reverse(view));
      assert(result.begin_ == buf);
      assert(result.end_ == buf + 3);
    }
  }

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

    // Check SFINAE friendliness
    {
      struct NotABidirRange { };
      static_assert(!std::is_invocable_v<decltype(std::views::reverse)>);
      static_assert(!std::is_invocable_v<decltype(std::views::reverse), NotABidirRange>);
      static_assert( CanBePiped<BidirRange,     decltype(std::views::reverse)>);
      static_assert( CanBePiped<BidirRange&,    decltype(std::views::reverse)>);
      static_assert(!CanBePiped<NotABidirRange, decltype(std::views::reverse)>);
    }
  }

  {
    static_assert(std::same_as<decltype(std::views::reverse), decltype(std::ranges::views::reverse)>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
