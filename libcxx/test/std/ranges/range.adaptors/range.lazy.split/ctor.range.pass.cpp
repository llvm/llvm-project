//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template <input_range Range>
//   requires constructible_from<View, views::all_t<Range>> &&
//             constructible_from<Pattern, single_view<range_value_t<Range>>>
// constexpr lazy_split_view(Range&& r, range_value_t<Range> e); // explicit since C++23

#include <ranges>

#include <cassert>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "test_convertible.h"
#include "test_macros.h"
#include "types.h"

struct ElementWithCounting {
  int* times_copied = nullptr;
  int* times_moved  = nullptr;

  constexpr ElementWithCounting(int& copies_ctr, int& moves_ctr) : times_copied(&copies_ctr), times_moved(&moves_ctr) {}

  constexpr ElementWithCounting(const ElementWithCounting& rhs)
      : times_copied(rhs.times_copied), times_moved(rhs.times_moved) {
    ++(*times_copied);
  }
  constexpr ElementWithCounting(ElementWithCounting&& rhs)
      : times_copied(rhs.times_copied), times_moved(rhs.times_moved) {
    ++(*times_moved);
  }

  constexpr bool operator==(const ElementWithCounting&) const { return true; }
};

struct RangeWithCounting {
  using value_type = ElementWithCounting;

  int* times_copied = nullptr;
  int* times_moved  = nullptr;

  constexpr RangeWithCounting(int& copies_ctr, int& moves_ctr) : times_copied(&copies_ctr), times_moved(&moves_ctr) {}

  constexpr RangeWithCounting(const RangeWithCounting& rhs)
      : times_copied(rhs.times_copied), times_moved(rhs.times_moved) {
    ++(*times_copied);
  }
  constexpr RangeWithCounting(RangeWithCounting&& rhs) : times_copied(rhs.times_copied), times_moved(rhs.times_moved) {
    ++(*times_moved);
  }

  constexpr const ElementWithCounting* begin() const { return nullptr; }
  constexpr const ElementWithCounting* end() const { return nullptr; }

  constexpr RangeWithCounting& operator=(const RangeWithCounting&) = default;
  constexpr RangeWithCounting& operator=(RangeWithCounting&&)      = default;
  constexpr bool operator==(const RangeWithCounting&) const { return true; }
};
static_assert(std::ranges::forward_range<RangeWithCounting>);
static_assert(!std::ranges::view<RangeWithCounting>);

struct StrView : std::ranges::view_base {
  std::string_view buffer_;
  constexpr explicit StrView() = default;
  constexpr StrView(const char* ptr) : buffer_(ptr) {}
  // Intentionally don't forward to range constructor for std::string_view since
  // this test needs to work on C++20 as well and the range constructor is only for
  // C++23 and later.
  template <std::ranges::range R>
  constexpr StrView(R&& r) : buffer_(r.begin(), r.end()) {}
  constexpr std::string_view::const_iterator begin() const { return buffer_.begin(); }
  constexpr std::string_view::const_iterator end() const { return buffer_.end(); }
  constexpr bool operator==(const StrView& rhs) const { return buffer_ == rhs.buffer_; }
};
static_assert(std::ranges::random_access_range<StrView>);
static_assert(std::ranges::view<StrView>);
static_assert(std::is_copy_constructible_v<StrView>);

// SFINAE tests.

#if TEST_STD_VER >= 23

static_assert(
    !test_convertible<std::ranges::lazy_split_view<StrView, StrView>, StrView, std::ranges::range_value_t<StrView>>(),
    "This constructor must be explicit");

#else

static_assert(
    test_convertible<std::ranges::lazy_split_view<StrView, StrView>, StrView, std::ranges::range_value_t<StrView>>(),
    "This constructor must not be explicit");

#endif // TEST_STD_VER >= 23

constexpr bool test() {
  {
    using V = std::ranges::lazy_split_view<StrView, StrView>;

    // Calling the constructor with `(std::string, range_value_t)`.
    {
      std::string input;
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
    using Range   = RangeWithCounting;
    using Element = ElementWithCounting;
    using Pattern = std::ranges::single_view<Element>;

    // Arguments are lvalues.
    {
      using View = std::ranges::ref_view<Range>;

      int range_copied = 0, range_moved = 0, element_copied = 0, element_moved = 0;
      Range range(range_copied, range_moved);
      Element element(element_copied, element_moved);

      std::ranges::lazy_split_view<View, Pattern> v(range, element);
      assert(range_copied == 0);   // `ref_view` does neither copy...
      assert(range_moved == 0);    // ...nor move the element.
      assert(element_copied == 1); // The element is copied into the argument...
#ifndef TEST_COMPILER_GCC
      //https://gcc.gnu.org/bugzilla/show_bug.cgi?id=98995
      assert(element_moved == 1); // ...and moved into the member variable.
#endif
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
#ifndef TEST_COMPILER_GCC
      //https://gcc.gnu.org/bugzilla/show_bug.cgi?id=98995
      assert(element_moved == 1);
#endif
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
