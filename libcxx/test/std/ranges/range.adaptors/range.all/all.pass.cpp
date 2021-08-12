//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// std::views::all;

#include <ranges>

#include <cassert>
#include "test_macros.h"
#include "test_iterators.h"

int globalBuff[8];

template<bool IsNoexcept>
struct View : std::ranges::view_base {
  int start_ = 0;
  explicit View() noexcept(IsNoexcept) = default;
  constexpr explicit View(int start) : start_(start) {}
  View(View&&) noexcept(IsNoexcept) = default;
  View& operator=(View&&) noexcept(IsNoexcept) = default;
  friend constexpr int* begin(View& view) { return globalBuff + view.start_; }
  friend constexpr int* begin(View const& view) { return globalBuff + view.start_; }
  friend constexpr int* end(View&) { return globalBuff + 8; }
  friend constexpr int* end(View const&) { return globalBuff + 8; }
};
static_assert(std::ranges::view<View<true>>);
static_assert(std::ranges::view<View<false>>);

template<bool IsNoexcept>
struct CopyableView : std::ranges::view_base {
  int start_ = 0;
  explicit CopyableView() noexcept(IsNoexcept) = default;
  CopyableView(CopyableView const&) noexcept(IsNoexcept) = default;
  CopyableView& operator=(CopyableView const&) noexcept(IsNoexcept) = default;
  constexpr explicit CopyableView(int start) noexcept : start_(start) {}
  friend constexpr int* begin(CopyableView& view) { return globalBuff + view.start_; }
  friend constexpr int* begin(CopyableView const& view) { return globalBuff + view.start_; }
  friend constexpr int* end(CopyableView&) { return globalBuff + 8; }
  friend constexpr int* end(CopyableView const&) { return globalBuff + 8; }
};
static_assert(std::ranges::view<CopyableView<true>>);
static_assert(std::ranges::view<CopyableView<false>>);

struct Range {
  int start_;
  constexpr explicit Range(int start) noexcept : start_(start) {}
  friend constexpr int* begin(Range const& range) { return globalBuff + range.start_; }
  friend constexpr int* begin(Range& range) { return globalBuff + range.start_; }
  friend constexpr int* end(Range const&) { return globalBuff + 8; }
  friend constexpr int* end(Range&) { return globalBuff + 8; }
};

struct BorrowableRange {
  int start_;
  constexpr explicit BorrowableRange(int start) noexcept : start_(start) {}
  friend constexpr int* begin(BorrowableRange const& range) { return globalBuff + range.start_; }
  friend constexpr int* begin(BorrowableRange& range) { return globalBuff + range.start_; }
  friend constexpr int* end(BorrowableRange const&) { return globalBuff + 8; }
  friend constexpr int* end(BorrowableRange&) { return globalBuff + 8; }
};
template<>
inline constexpr bool std::ranges::enable_borrowed_range<BorrowableRange> = true;

struct RandomAccessRange {
  struct sentinel {
    friend constexpr bool operator==(sentinel, const random_access_iterator<int*> rai) { return rai.base() == globalBuff + 8; }
    friend constexpr std::ptrdiff_t operator-(sentinel, random_access_iterator<int*>) { return -8; }
    friend constexpr std::ptrdiff_t operator-(random_access_iterator<int*>, sentinel) { return 8; }
  };

  constexpr random_access_iterator<int*> begin() { return random_access_iterator<int*>{globalBuff}; }
  constexpr sentinel end() { return {}; }
};
template<>
inline constexpr bool std::ranges::enable_borrowed_range<RandomAccessRange> = true;

constexpr bool test() {
  {
    ASSERT_SAME_TYPE(decltype(std::views::all(View<true>())), View<true>);
    static_assert(noexcept(std::views::all(View<true>())));
    static_assert(!noexcept(std::views::all(View<false>())));

    auto viewCopy = std::views::all(View<true>(2));
    ASSERT_SAME_TYPE(decltype(viewCopy), View<true>);
    assert(std::ranges::begin(viewCopy) == globalBuff + 2);
    assert(std::ranges::end(viewCopy) == globalBuff + 8);
  }

  {
    ASSERT_SAME_TYPE(decltype(std::views::all(std::declval<const CopyableView<true>&>())), CopyableView<true>);
    static_assert(noexcept(std::views::all(CopyableView<true>())));
    static_assert(!noexcept(std::views::all(CopyableView<false>())));

    CopyableView<true> view(2);
    auto viewCopy = std::views::all(view);
    ASSERT_SAME_TYPE(decltype(viewCopy), CopyableView<true>);
    assert(std::ranges::begin(viewCopy) == globalBuff + 2);
    assert(std::ranges::end(viewCopy) == globalBuff + 8);
  }

  {
    Range range(2);
    auto ref = std::views::all(range);
    ASSERT_SAME_TYPE(decltype(ref), std::ranges::ref_view<Range>);
    assert(std::ranges::begin(ref) == globalBuff + 2);
    assert(std::ranges::end(ref) == globalBuff + 8);

    static_assert(!std::is_invocable_v<decltype(std::views::all), Range>);
  }

  {
    const Range range(2);
    auto ref = std::views::all(range);
    static_assert(!noexcept(std::views::all(range)));
    ASSERT_SAME_TYPE(decltype(ref), std::ranges::ref_view<const Range>);
    assert(std::ranges::begin(ref) == globalBuff + 2);
    assert(std::ranges::end(ref) == globalBuff + 8);
  }

  {
    auto subrange = std::views::all(BorrowableRange(2));
    static_assert(!noexcept(std::views::all(BorrowableRange(2))));
    ASSERT_SAME_TYPE(decltype(subrange), std::ranges::subrange<int*>);
    assert(std::ranges::begin(subrange) == globalBuff + 2);
    assert(std::ranges::end(subrange) == globalBuff + 8);
  }

  {
    auto subrange = std::views::all(RandomAccessRange());
    ASSERT_SAME_TYPE(decltype(subrange),
                     std::ranges::subrange<random_access_iterator<int*>, RandomAccessRange::sentinel>);
    assert(std::ranges::begin(subrange).base() == globalBuff);
    assert(std::ranges::end(subrange) == std::ranges::begin(subrange) + 8);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
