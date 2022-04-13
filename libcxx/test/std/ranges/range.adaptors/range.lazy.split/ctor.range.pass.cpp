//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// template <input_range Range>
//   requires constructible_from<View, views::all_t<Range>> &&
//             constructible_from<Pattern, single_view<range_value_t<Range>>>
// constexpr lazy_split_view(Range&& r, range_value_t<Range> e);

#include <ranges>

#include <cassert>
#include <string_view>
#include <type_traits>
#include <utility>
#include "small_string.h"
#include "types.h"

struct ElementWithCounting {
  int* times_copied = nullptr;
  int* times_moved = nullptr;

  constexpr ElementWithCounting(int& copies_ctr, int& moves_ctr) : times_copied(&copies_ctr), times_moved(&moves_ctr) {}

  constexpr ElementWithCounting(const ElementWithCounting& rhs)
      : times_copied(rhs.times_copied)
      , times_moved(rhs.times_moved) {
    ++(*times_copied);
  }
  constexpr ElementWithCounting(ElementWithCounting&& rhs)
      : times_copied(rhs.times_copied)
      , times_moved(rhs.times_moved) {
    ++(*times_moved);
  }

  constexpr bool operator==(const ElementWithCounting&) const { return true; }
};

struct RangeWithCounting {
  using value_type = ElementWithCounting;

  int* times_copied = nullptr;
  int* times_moved = nullptr;

  constexpr RangeWithCounting(int& copies_ctr, int& moves_ctr) : times_copied(&copies_ctr), times_moved(&moves_ctr) {}

  constexpr RangeWithCounting(const RangeWithCounting& rhs)
    : times_copied(rhs.times_copied)
    , times_moved(rhs.times_moved) {
    ++(*times_copied);
  }
  constexpr RangeWithCounting(RangeWithCounting&& rhs)
    : times_copied(rhs.times_copied)
    , times_moved(rhs.times_moved) {
    ++(*times_moved);
  }

  constexpr const ElementWithCounting* begin() const { return nullptr; }
  constexpr const ElementWithCounting* end() const { return nullptr; }

  constexpr RangeWithCounting& operator=(const RangeWithCounting&) = default;
  constexpr RangeWithCounting& operator=(RangeWithCounting&&) = default;
  constexpr bool operator==(const RangeWithCounting&) const { return true; }
};
static_assert( std::ranges::forward_range<RangeWithCounting>);
static_assert(!std::ranges::view<RangeWithCounting>);

struct StrRange {
  SmallString buffer_;
  constexpr explicit StrRange() = default;
  constexpr StrRange(const char* ptr) : buffer_(ptr) {}
  constexpr const char* begin() const { return buffer_.begin(); }
  constexpr const char* end() const { return buffer_.end(); }
  constexpr bool operator==(const StrRange& rhs) const { return buffer_ == rhs.buffer_; }
};
static_assert( std::ranges::random_access_range<StrRange>);
static_assert(!std::ranges::view<StrRange>);
static_assert( std::is_copy_constructible_v<StrRange>);

struct StrView : std::ranges::view_base {
  SmallString buffer_;
  constexpr explicit StrView() = default;
  constexpr StrView(const char* ptr) : buffer_(ptr) {}
  template <std::ranges::range R>
  constexpr StrView(R&& r) : buffer_(std::forward<R>(r)) {}
  constexpr const char* begin() const { return buffer_.begin(); }
  constexpr const char* end() const { return buffer_.end(); }
  constexpr bool operator==(const StrView& rhs) const { return buffer_ == rhs.buffer_; }
};
static_assert( std::ranges::random_access_range<StrView>);
static_assert( std::ranges::view<StrView>);
static_assert( std::is_copy_constructible_v<StrView>);

constexpr bool test() {
  {
    using V = std::ranges::lazy_split_view<StrView, StrView>;

    // Calling the constructor with `(StrRange, range_value_t)`.
    {
      StrRange input;
      V v(input, ' ');
      assert(v.base() == input);
    }

    // Calling the constructor with `(StrView, range_value_t)`.
    {
      StrView input("abc def");
      V v(input, ' ');
      assert(v.base() == input);
    }

    struct Empty {};
    static_assert(!std::is_constructible_v<V, Empty, std::string_view>);
    static_assert(!std::is_constructible_v<V, std::string_view, Empty>);
  }

  // Make sure the arguments are moved, not copied.
  {
    using Range = RangeWithCounting;
    using Element = ElementWithCounting;
    // TODO(varconst): use `views::single` once it's implemented.
    using Pattern = std::ranges::single_view<Element>;

    // Arguments are lvalues.
    {
      using View = std::ranges::ref_view<Range>;

      int range_copied = 0, range_moved = 0, element_copied = 0, element_moved = 0;
      Range range(range_copied, range_moved);
      Element element(element_copied, element_moved);

      std::ranges::lazy_split_view<View, Pattern> v(range, element);
      assert(range_copied == 0); // `ref_view` does neither copy...
      assert(range_moved == 0); // ...nor move the element.
      assert(element_copied == 1); // The element is copied into the argument...
      assert(element_moved == 1); // ...and moved into the member variable.
    }

    // Arguments are rvalues.
    {
      using View = std::ranges::owning_view<Range>;

      int range_copied = 0, range_moved = 0, element_copied = 0, element_moved = 0;
      std::ranges::lazy_split_view<View, Pattern> v(
          Range(range_copied, range_moved), Element(element_copied, element_moved));
      assert(range_copied == 0);
      assert(range_moved == 1); // `owning_view` moves the given argument.
      assert(element_copied == 0);
      assert(element_moved == 1);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
