//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <ranges>

//  template<input_range V>
//    requires view<V>
//  class as_input_view : public view_interface<as_input_view<V>>

// [range.to.input.overview]

#include <cassert>
#include <concepts>
#include <functional>
#include <ranges>
#include <vector>

#include "test_iterators.h"
#include "test_range.h"

struct NonView {};

static_assert(std::default_initializable<NonView>);

struct DefaultInitializableView : std::ranges::view_base {
  int i_;

  int* begin();
  int* end();
};

static_assert(std::default_initializable<DefaultInitializableView>);
static_assert(std::ranges::common_range<DefaultInitializableView>);
static_assert(std::ranges::input_range<DefaultInitializableView>);

struct CommonView : std::ranges::view_base {
  int i_;

  constexpr forward_iterator<int*> begin() { return forward_iterator<int*>(&i_); }
  constexpr forward_iterator<int*> end() { return begin(); }
};

static_assert(std::ranges::common_range<CommonView>);
static_assert(std::ranges::forward_range<CommonView>);
static_assert(std::ranges::input_range<CommonView>);

struct NonCommonView : std::ranges::view_base {
  int i_;

  constexpr forward_iterator<int*> begin() { return forward_iterator<int*>(&i_); }
  constexpr sentinel_wrapper<forward_iterator<int*>> end() { return sentinel_wrapper<forward_iterator<int*>>(begin()); }
};

static_assert(!std::ranges::common_range<NonCommonView>);
static_assert(std::ranges::forward_range<NonCommonView>);
static_assert(std::ranges::input_range<NonCommonView>);
static_assert(
    std::derived_from< typename std::iterator_traits<std::ranges::iterator_t<NonCommonView>>::iterator_category,
                       std::input_iterator_tag>);

// Check that the `as_input` adaptor can be used with a view.

static_assert(!std::is_invocable_v<decltype(std::views::as_input)>);
static_assert(!std::is_invocable_v<decltype(std::views::as_input), NonView>);
static_assert(std::is_invocable_v<decltype(std::views::as_input), DefaultInitializableView>);
static_assert(std::is_invocable_v<decltype(std::views::as_input), CommonView>);
static_assert(std::is_invocable_v<decltype(std::views::as_input), NonCommonView>);

static_assert(!CanBePiped<NonView, decltype(std::views::as_input)>);
static_assert(CanBePiped<DefaultInitializableView&, decltype(std::views::as_input)>);
static_assert(CanBePiped<CommonView&, decltype(std::views::as_input)>);
static_assert(CanBePiped<NonCommonView&, decltype(std::views::as_input)>);

constexpr bool test() {
  // Sameness
  {
    static_assert(std::is_same_v<decltype(std::views::as_input), decltype(std::ranges::views::as_input)>);
    assert(std::addressof(std::views::as_input) == std::addressof(std::ranges::views::as_input));
  }

  { // view | views::as_input

    DefaultInitializableView view{{}, 94};
    std::same_as<std::ranges::as_input_view<DefaultInitializableView>> decltype(auto) v =
        view | std::views::as_input | std::views::as_input;
    assert(v.base().i_ == 94);

    static_assert(!std::ranges::common_range<decltype(v)>);
    static_assert(!std::ranges::forward_range<decltype(v)>);
    static_assert(std::ranges::input_range<decltype(v)>);
  }

  { // view | views::as_input
    {
      DefaultInitializableView view{{}, 94};
      std::same_as<std::ranges::as_input_view<DefaultInitializableView>> decltype(auto) v = view | std::views::as_input;
      assert(v.base().i_ == 94);

      static_assert(!std::ranges::common_range<decltype(v)>);
      static_assert(!std::ranges::forward_range<decltype(v)>);
      static_assert(std::ranges::input_range<decltype(v)>);
    }
    {
      CommonView view{{}, 94};
      std::same_as<std::ranges::as_input_view<CommonView>> decltype(auto) v = view | std::views::as_input;
      assert(v.base().i_ == 94);

      static_assert(!std::ranges::common_range<decltype(v)>);
      static_assert(!std::ranges::forward_range<decltype(v)>);
      static_assert(std::ranges::input_range<decltype(v)>);
    }
    {
      NonCommonView view{{}, 94};
      std::same_as<std::ranges::as_input_view<NonCommonView>> decltype(auto) v = view | std::views::as_input;
      assert(v.base().i_ == 94);

      static_assert(!std::ranges::common_range<decltype(v)>);
      static_assert(!std::ranges::forward_range<decltype(v)>);
      static_assert(std::ranges::input_range<decltype(v)>);
    }
  }

  { // adaptor | views::as_input
    {
      DefaultInitializableView view{{}, 94};
      const auto partial = std::views::transform(std::identity{}) | std::views::as_input;
      std::same_as<std::ranges::as_input_view<
          std::ranges::transform_view<DefaultInitializableView, std::identity>>> decltype(auto) v = partial(view);
      assert(v.base().base().i_ == 94);

      static_assert(!std::ranges::common_range<decltype(v)>);
      static_assert(!std::ranges::forward_range<decltype(v)>);
      static_assert(std::ranges::input_range<decltype(v)>);
    }
    {
      CommonView view{{}, 94};
      const auto partial = std::views::transform(std::identity{}) | std::views::as_input;
      std::same_as<std::ranges::as_input_view< std::ranges::transform_view<CommonView, std::identity>>> decltype(auto)
          v = partial(view);
      assert(v.base().base().i_ == 94);

      static_assert(!std::ranges::common_range<decltype(v)>);
      static_assert(!std::ranges::forward_range<decltype(v)>);
      static_assert(std::ranges::input_range<decltype(v)>);
    }
    {
      NonCommonView view{{}, 94};
      const auto partial = std::views::transform(std::identity{}) | std::views::as_input;
      std::same_as<
          std::ranges::as_input_view< std::ranges::transform_view<NonCommonView, std::identity>>> decltype(auto) v =
          partial(view);
      assert(v.base().base().i_ == 94);

      static_assert(!std::ranges::common_range<decltype(v)>);
      static_assert(!std::ranges::forward_range<decltype(v)>);
      static_assert(std::ranges::input_range<decltype(v)>);
    }
  }

  { // views::as_input | adaptor
    {
      DefaultInitializableView view{{}, 94};
      const auto partial = std::views::as_input | std::views::transform(std::identity{});
      std::same_as<std::ranges::transform_view<std::ranges::as_input_view<DefaultInitializableView>,
                                               std::identity>> decltype(auto) v = partial(view);
      assert(v.base().base().i_ == 94);

      static_assert(std::ranges::input_range<decltype(v)>);
      static_assert(!std::ranges::forward_range<decltype(v)>);
      static_assert(!std::ranges::common_range<decltype(v)>);
    }
    {
      CommonView view{{}, 94};
      const auto partial = std::views::as_input | std::views::transform(std::identity{});
      std::same_as<std::ranges::transform_view<std::ranges::as_input_view<CommonView>, std::identity>> decltype(auto)
          v = partial(view);
      assert(v.base().base().i_ == 94);

      static_assert(std::ranges::input_range<decltype(v)>);
      static_assert(!std::ranges::forward_range<decltype(v)>);
      static_assert(!std::ranges::common_range<decltype(v)>);
    }
    {
      NonCommonView view{{}, 94};
      const auto partial = std::views::as_input | std::views::transform(std::identity{});
      std::same_as<
          std::ranges::transform_view<std::ranges::as_input_view<NonCommonView>, std::identity>> decltype(auto) v =
          partial(view);
      assert(v.base().base().i_ == 94);

      static_assert(std::ranges::input_range<decltype(v)>);
      static_assert(!std::ranges::forward_range<decltype(v)>);
      static_assert(!std::ranges::common_range<decltype(v)>);
    }
  }

  { // views::as_input | views::all
    {
      DefaultInitializableView view{{}, 94};
      std::same_as<std::ranges::as_input_view<DefaultInitializableView>> decltype(auto) v =
          std::views::all(view) | std::views::as_input;
      assert(v.base().i_ == 94);

      static_assert(std::ranges::input_range<decltype(v)>);
      static_assert(!std::ranges::forward_range<decltype(v)>);
      static_assert(!std::ranges::common_range<decltype(v)>);
    }
    {
      CommonView view{{}, 94};
      std::same_as<std::ranges::as_input_view<CommonView>> decltype(auto) v =
          std::views::all(view) | std::views::as_input;
      assert(v.base().i_ == 94);

      static_assert(std::ranges::input_range<decltype(v)>);
      static_assert(!std::ranges::forward_range<decltype(v)>);
      static_assert(!std::ranges::common_range<decltype(v)>);
    }
    {
      NonCommonView view{{}, 94};
      std::same_as<std::ranges::as_input_view<NonCommonView>> decltype(auto) v =
          std::views::all(view) | std::views::as_input;
      assert(v.base().i_ == 94);

      static_assert(std::ranges::input_range<decltype(v)>);
      static_assert(!std::ranges::forward_range<decltype(v)>);
      static_assert(!std::ranges::common_range<decltype(v)>);
    }
  }

  { // views::as_input | views::all_t
    {
      DefaultInitializableView view{{}, 94};
      std::same_as<std::ranges::as_input_view<DefaultInitializableView>> decltype(auto) v =
          std::views::all_t<DefaultInitializableView>(view) | std::views::as_input;
      assert(v.base().i_ == 94);

      static_assert(std::ranges::input_range<decltype(v)>);
      static_assert(!std::ranges::forward_range<decltype(v)>);
      static_assert(!std::ranges::common_range<decltype(v)>);
    }
    {
      CommonView view{{}, 94};
      std::same_as<std::ranges::as_input_view<CommonView>> decltype(auto) v =
          std::views::all_t<CommonView>(view) | std::views::as_input;
      assert(v.base().i_ == 94);

      static_assert(std::ranges::input_range<decltype(v)>);
      static_assert(!std::ranges::forward_range<decltype(v)>);
      static_assert(!std::ranges::common_range<decltype(v)>);
    }
    {
      NonCommonView view{{}, 94};
      std::same_as<std::ranges::as_input_view<NonCommonView>> decltype(auto) v =
          std::views::all_t<NonCommonView>(view) | std::views::as_input;
      assert(v.base().i_ == 94);

      static_assert(std::ranges::input_range<decltype(v)>);
      static_assert(!std::ranges::forward_range<decltype(v)>);
      static_assert(!std::ranges::common_range<decltype(v)>);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
