//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <ranges>

// class cache_latest_view

#include <cassert>
#include <concepts>
#include <ranges>

#include "test_iterators.h"
#include "test_range.h"

static_assert(std::is_same_v<decltype(std::views::cache_latest), decltype(std::ranges::views::cache_latest)>);

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

// Check that the `cache_latest` adaptor can be used with a view.

static_assert(!std::is_invocable_v<decltype(std::views::cache_latest)>);
static_assert(!std::is_invocable_v<decltype(std::views::cache_latest), NonView>);
static_assert(std::is_invocable_v<decltype(std::views::cache_latest), DefaultInitializableView>);
static_assert(std::is_invocable_v<decltype(std::views::cache_latest), CommonView>);
static_assert(std::is_invocable_v<decltype(std::views::cache_latest), NonCommonView>);

static_assert(!CanBePiped<NonView, decltype(std::views::cache_latest)>);
static_assert(CanBePiped<DefaultInitializableView&, decltype(std::views::cache_latest)>);
static_assert(CanBePiped<CommonView&, decltype(std::views::cache_latest)>);
static_assert(CanBePiped<NonCommonView&, decltype(std::views::cache_latest)>);

constexpr bool test() {
  { // view | views::cache_latest
    {
      DefaultInitializableView view{{}, 94};
      std::same_as<std::ranges::cache_latest_view<DefaultInitializableView>> decltype(auto) v =
          view | std::views::cache_latest;
      assert(v.base().i_ == 94);

      static_assert(!std::ranges::common_range<decltype(v)>);
      static_assert(!std::ranges::forward_range<decltype(v)>);
      static_assert(std::ranges::input_range<decltype(v)>);
    }
    {
      CommonView view{{}, 94};
      std::same_as<std::ranges::cache_latest_view<CommonView>> decltype(auto) v = view | std::views::cache_latest;
      assert(v.base().i_ == 94);

      static_assert(!std::ranges::common_range<decltype(v)>);
      static_assert(!std::ranges::forward_range<decltype(v)>);
      static_assert(std::ranges::input_range<decltype(v)>);
    }
    {
      NonCommonView view{{}, 94};
      std::same_as<std::ranges::cache_latest_view<NonCommonView>> decltype(auto) v = view | std::views::cache_latest;
      assert(v.base().i_ == 94);

      static_assert(!std::ranges::common_range<decltype(v)>);
      static_assert(!std::ranges::forward_range<decltype(v)>);
      static_assert(std::ranges::input_range<decltype(v)>);
    }
  }

  { // adaptor | views::cache_latest
    {
      DefaultInitializableView view{{}, 94};
      const auto partial = std::views::transform(std::identity{}) | std::views::cache_latest;
      std::same_as<std::ranges::cache_latest_view<
          std::ranges::transform_view<DefaultInitializableView, std::identity>>> decltype(auto) v = partial(view);
      assert(v.base().base().i_ == 94);

      static_assert(!std::ranges::common_range<decltype(v)>);
      static_assert(!std::ranges::forward_range<decltype(v)>);
      static_assert(std::ranges::input_range<decltype(v)>);
    }
    {
      CommonView view{{}, 94};
      const auto partial = std::views::transform(std::identity{}) | std::views::cache_latest;
      std::same_as<
          std::ranges::cache_latest_view< std::ranges::transform_view<CommonView, std::identity>>> decltype(auto) v =
          partial(view);
      assert(v.base().base().i_ == 94);

      static_assert(!std::ranges::common_range<decltype(v)>);
      static_assert(!std::ranges::forward_range<decltype(v)>);
      static_assert(std::ranges::input_range<decltype(v)>);
    }
    {
      NonCommonView view{{}, 94};
      const auto partial = std::views::transform(std::identity{}) | std::views::cache_latest;
      std::same_as<
          std::ranges::cache_latest_view< std::ranges::transform_view<NonCommonView, std::identity>>> decltype(auto) v =
          partial(view);
      assert(v.base().base().i_ == 94);

      static_assert(!std::ranges::common_range<decltype(v)>);
      static_assert(!std::ranges::forward_range<decltype(v)>);
      static_assert(std::ranges::input_range<decltype(v)>);
    }
  }

  { // views::cache_latest | adaptor
    {
      DefaultInitializableView view{{}, 94};
      const auto partial = std::views::cache_latest | std::views::transform(std::identity{});
      std::same_as<std::ranges::transform_view<std::ranges::cache_latest_view<DefaultInitializableView>,
                                               std::identity>> decltype(auto) v = partial(view);
      assert(v.base().base().i_ == 94);

      static_assert(std::ranges::input_range<decltype(v)>);
      static_assert(!std::ranges::forward_range<decltype(v)>);
      static_assert(!std::ranges::common_range<decltype(v)>);
    }
    {
      CommonView view{{}, 94};
      const auto partial = std::views::cache_latest | std::views::transform(std::identity{});
      std::same_as<
          std::ranges::transform_view<std::ranges::cache_latest_view<CommonView>, std::identity>> decltype(auto) v =
          partial(view);
      assert(v.base().base().i_ == 94);

      static_assert(std::ranges::input_range<decltype(v)>);
      static_assert(!std::ranges::forward_range<decltype(v)>);
      static_assert(!std::ranges::common_range<decltype(v)>);
    }
    {
      NonCommonView view{{}, 94};
      const auto partial = std::views::cache_latest | std::views::transform(std::identity{});
      std::same_as<
          std::ranges::transform_view<std::ranges::cache_latest_view<NonCommonView>, std::identity>> decltype(auto) v =
          partial(view);
      assert(v.base().base().i_ == 94);

      static_assert(std::ranges::input_range<decltype(v)>);
      static_assert(!std::ranges::forward_range<decltype(v)>);
      static_assert(!std::ranges::common_range<decltype(v)>);
    }
  }

  { // views::cache_latest | views::all
    {
      DefaultInitializableView view{{}, 94};
      std::same_as<std::ranges::cache_latest_view<DefaultInitializableView>> decltype(auto) v =
          std::views::all(view) | std::views::cache_latest;
      assert(v.base().i_ == 94);

      static_assert(std::ranges::input_range<decltype(v)>);
      static_assert(!std::ranges::forward_range<decltype(v)>);
      static_assert(!std::ranges::common_range<decltype(v)>);
    }
    {
      CommonView view{{}, 94};
      std::same_as<std::ranges::cache_latest_view<CommonView>> decltype(auto) v =
          std::views::all(view) | std::views::cache_latest;
      assert(v.base().i_ == 94);

      static_assert(std::ranges::input_range<decltype(v)>);
      static_assert(!std::ranges::forward_range<decltype(v)>);
      static_assert(!std::ranges::common_range<decltype(v)>);
    }
    {
      NonCommonView view{{}, 94};
      std::same_as<std::ranges::cache_latest_view<NonCommonView>> decltype(auto) v =
          std::views::all(view) | std::views::cache_latest;
      assert(v.base().i_ == 94);

      static_assert(std::ranges::input_range<decltype(v)>);
      static_assert(!std::ranges::forward_range<decltype(v)>);
      static_assert(!std::ranges::common_range<decltype(v)>);
    }
  }

  { // views::cache_latest | views::all_t
    {
      DefaultInitializableView view{{}, 94};
      std::same_as<std::ranges::cache_latest_view<DefaultInitializableView>> decltype(auto) v =
          std::views::all_t<DefaultInitializableView>(view) | std::views::cache_latest;
      assert(v.base().i_ == 94);

      static_assert(std::ranges::input_range<decltype(v)>);
      static_assert(!std::ranges::forward_range<decltype(v)>);
      static_assert(!std::ranges::common_range<decltype(v)>);
    }
    {
      CommonView view{{}, 94};
      std::same_as<std::ranges::cache_latest_view<CommonView>> decltype(auto) v =
          std::views::all_t<CommonView>(view) | std::views::cache_latest;
      assert(v.base().i_ == 94);

      static_assert(std::ranges::input_range<decltype(v)>);
      static_assert(!std::ranges::forward_range<decltype(v)>);
      static_assert(!std::ranges::common_range<decltype(v)>);
    }
    {
      NonCommonView view{{}, 94};
      std::same_as<std::ranges::cache_latest_view<NonCommonView>> decltype(auto) v =
          std::views::all_t<NonCommonView>(view) | std::views::cache_latest;
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
