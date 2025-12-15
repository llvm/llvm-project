//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr View base() const& requires copy_constructible<View>;
// constexpr View base() &&;

#include <ranges>

#include <cassert>
#include <concepts>
#include <utility>

struct Range : std::ranges::view_base {
  template <std::size_t N>
  constexpr explicit Range(int (&buffer)[N]) : begin_(&buffer[0]), end_(&buffer[0] + N) {}
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

template <std::size_t N>
constexpr void test() {
  int buff[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

  // Check the const& overload
  {
    Range range(buff);
    auto view = range | std::views::adjacent<N>;

    std::same_as<Range> decltype(auto) result = view.base();
    assert(result.wasCopyInitialized);
    assert(result.begin() == buff);
    assert(result.end() == buff + 9);
  }

  // Check the && overload
  {
    Range range(buff);
    auto view                                 = range | std::views::adjacent<N>;
    std::same_as<Range> decltype(auto) result = std::move(view).base();
    assert(result.wasMoveInitialized);
    assert(result.begin() == buff);
    assert(result.end() == buff + 9);
  }

  // Ensure the const& overload is not considered when the base is not copy-constructible
  {
    static_assert(!CanCallBaseOn<std::ranges::adjacent_view<NonCopyableRange, N> const&>);
    static_assert(!CanCallBaseOn<std::ranges::adjacent_view<NonCopyableRange, N>&>);
    static_assert(!CanCallBaseOn<std::ranges::adjacent_view<NonCopyableRange, N> const&&>);
    static_assert(CanCallBaseOn<std::ranges::adjacent_view<NonCopyableRange, N>&&>);
    static_assert(CanCallBaseOn<std::ranges::adjacent_view<NonCopyableRange, N>>);
  }
}

constexpr bool test() {
  test<1>();
  test<2>();
  test<3>();
  test<4>();
  test<5>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
