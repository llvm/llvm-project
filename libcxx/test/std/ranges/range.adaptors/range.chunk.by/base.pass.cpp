//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <ranges>

// constexpr View base() const& requires copy_constructible<View>;
// constexpr View base() &&;

#include <ranges>

#include <cassert>
#include <concepts>
#include <utility>

struct Range : std::ranges::view_base {
  constexpr explicit Range(int* b, int* e) : begin_(b), end_(e) {}
  constexpr Range(Range const& other) : begin_(other.begin_), end_(other.end_), wasCopyInitialized(true) {}
  constexpr Range(Range&& other) : begin_(other.begin_), end_(other.end_), wasMoveInitialized(true) {}
  Range& operator=(Range const&) = default;
  Range& operator=(Range&&)      = default;
  constexpr int* begin() const { return begin_; }
  constexpr int* end() const { return end_; }

  int* begin_;
  int* end_;
  bool wasCopyInitialized = false;
  bool wasMoveInitialized = false;
};

static_assert(std::ranges::view<Range>);
static_assert(std::ranges::forward_range<Range>);

struct Pred {
  bool operator()(int, int) const;
};

struct NonCopyableRange : std::ranges::view_base {
  explicit NonCopyableRange(int*, int*);
  NonCopyableRange(NonCopyableRange const&)            = delete;
  NonCopyableRange(NonCopyableRange&&)                 = default;
  NonCopyableRange& operator=(NonCopyableRange const&) = default;
  NonCopyableRange& operator=(NonCopyableRange&&)      = default;
  int* begin() const;
  int* end() const;
};

static_assert(!std::copy_constructible<NonCopyableRange>);

template <typename T>
concept CanCallBaseOn = requires(T t) { std::forward<T>(t).base(); };

constexpr bool test() {
  int buff[] = {1, 2, 3, 4};

  // Check the const& overload
  {
    Range range(buff, buff + 4);
    std::ranges::chunk_by_view<Range, Pred> const view(range, Pred{});
    std::same_as<Range> decltype(auto) result = view.base();
    assert(result.wasCopyInitialized);
    assert(result.begin() == buff);
    assert(result.end() == buff + 4);
  }

  // Check the && overload
  {
    Range range(buff, buff + 4);
    std::ranges::chunk_by_view<Range, Pred> view(range, Pred{});
    std::same_as<Range> decltype(auto) result = std::move(view).base();
    assert(result.wasMoveInitialized);
    assert(result.begin() == buff);
    assert(result.end() == buff + 4);
  }

  // Ensure the const& overload is not considered when the base is not copy-constructible
  {
    static_assert(!CanCallBaseOn<std::ranges::chunk_by_view<NonCopyableRange, Pred> const&>);
    static_assert(!CanCallBaseOn<std::ranges::chunk_by_view<NonCopyableRange, Pred>&>);
    static_assert(!CanCallBaseOn<std::ranges::chunk_by_view<NonCopyableRange, Pred> const&&>);
    static_assert(CanCallBaseOn<std::ranges::chunk_by_view<NonCopyableRange, Pred>&&>);
    static_assert(CanCallBaseOn<std::ranges::chunk_by_view<NonCopyableRange, Pred>>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
