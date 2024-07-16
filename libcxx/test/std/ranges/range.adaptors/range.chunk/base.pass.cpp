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
  constexpr Range(Range const& other) : was_copy_initialized(true), begin_(other.begin_), end_(other.end_) {}
  constexpr Range(Range&& other) : was_move_initialized(true), begin_(other.begin_), end_(other.end_) {}
  Range& operator=(Range const&) = default;
  Range& operator=(Range&&)      = default;
  constexpr int* begin() const { return begin_; }
  constexpr int* end() const { return end_; }

  bool was_copy_initialized = false;
  bool was_move_initialized = false;

private:
  int* begin_;
  int* end_;
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
concept CanCallBaseOn = requires(T&& t) { std::forward<T>(t).base(); };

constexpr bool test() {
  constexpr int N = 4;
  int buf[N]      = {1, 2, 3, 4};

  // Check the const& overload
  {
    Range range(buf, buf + N);
    const std::ranges::chunk_view<Range> view(range, N);
    std::same_as<Range> decltype(auto) result = view.base();
    assert(result.was_copy_initialized);
    assert(result.begin() == buf);
    assert(result.end() == buf + N);
  }

  // Check the && overload
  {
    Range range(buf, buf + N);
    std::ranges::chunk_view<Range> view(range, N);
    std::same_as<Range> decltype(auto) result = std::move(view).base();
    assert(result.was_move_initialized);
    assert(result.begin() == buf);
    assert(result.end() == buf + N);
  }

  // Ensure the const& overloads are not considered when the base is not copy-constructible
  {
    static_assert(!CanCallBaseOn<const std::ranges::chunk_view<NonCopyableRange>&>);
    static_assert(!CanCallBaseOn<std::ranges::chunk_view<NonCopyableRange>&>);
    static_assert(!CanCallBaseOn<const std::ranges::chunk_view<NonCopyableRange>&&>);
    static_assert(CanCallBaseOn<std::ranges::chunk_view<NonCopyableRange>&&>);
    static_assert(CanCallBaseOn<std::ranges::chunk_view<NonCopyableRange>>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
