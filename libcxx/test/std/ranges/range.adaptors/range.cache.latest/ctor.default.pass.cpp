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

//     cache_latest_view() requires default_initializable<V> = default;

#include <cassert>
#include <concepts>
#include <ranges>

struct DefaultInitializableView : std::ranges::view_base {
  int i_ = 94;

  int* begin();
  int* end();
};

static_assert(std::default_initializable<DefaultInitializableView>);
static_assert(std::default_initializable<std::ranges::cache_latest_view<DefaultInitializableView>>);

struct NoDefaultInitializableView : std::ranges::view_base {
  NoDefaultInitializableView() = delete;

  int* begin();
  int* end();
};

static_assert(!std::default_initializable<NoDefaultInitializableView>);
static_assert(!std::default_initializable<std::ranges::cache_latest_view<NoDefaultInitializableView>>);

struct NoexceptView : std::ranges::view_base {
  NoexceptView() noexcept;

  int const* begin() const;
  int const* end() const;
};

static_assert(noexcept(std::ranges::cache_latest_view<NoexceptView>()));

struct NoNoexceptView : std::ranges::view_base {
  NoNoexceptView() noexcept(false);

  int const* begin() const;
  int const* end() const;
};

static_assert(!noexcept(std::ranges::cache_latest_view<NoNoexceptView>()));

constexpr bool test() {
  { // value-initialized (i.e., whether T() is well-formed).
    std::ranges::cache_latest_view<DefaultInitializableView> cache_latest_view{};
    assert(cache_latest_view.base().i_ == 94);
  }
  { // direct-list-initialized from an empty initializer list (i.e., whether T{} is well-formed).
    std::ranges::cache_latest_view<DefaultInitializableView> cache_latest_view = {};
    assert(cache_latest_view.base().i_ == 94);
  }
  { // default-initialized (i.e., whether T t; is well-formed).
    std::ranges::cache_latest_view<DefaultInitializableView> cache_latest_view;
    assert(cache_latest_view.base().i_ == 94);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
