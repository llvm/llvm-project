//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <ranges>

// class to_input_view

//    to_input_view() requires default_initializable<V> = default;

#include <cassert>
#include <concepts>
#include <ranges>

struct DefaultInitializableView : std::ranges::view_base {
  int i_ = 94;

  int* begin();
  int* end();
};

static_assert(std::default_initializable<DefaultInitializableView>);
static_assert(std::default_initializable<std::ranges::to_input_view<DefaultInitializableView>>);

struct NoDefaultInitializableView : std::ranges::view_base {
  NoDefaultInitializableView() = delete;

  int* begin();
  int* end();
};

static_assert(!std::default_initializable<NoDefaultInitializableView>);
static_assert(!std::default_initializable<std::ranges::to_input_view<NoDefaultInitializableView>>);

struct NoexceptView : std::ranges::view_base {
  NoexceptView() noexcept;

  int const* begin() const;
  int const* end() const;
};

static_assert(noexcept(std::ranges::to_input_view<NoexceptView>()));

struct NoNoexceptView : std::ranges::view_base {
  NoNoexceptView() noexcept(false);

  int const* begin() const;
  int const* end() const;
};

static_assert(!noexcept(std::ranges::to_input_view<NoNoexceptView>()));

constexpr bool test() {
  { // value-initialized (i.e., whether T() is well-formed).
    std::ranges::to_input_view<DefaultInitializableView> to_input_view{};
    assert(to_input_view.base().i_ == 94);
  }
  { // direct-list-initialized from an empty initializer list (i.e., whether T{} is well-formed).
    std::ranges::to_input_view<DefaultInitializableView> to_input_view = {};
    assert(to_input_view.base().i_ == 94);
  }
  { // default-initialized (i.e., whether T t; is well-formed).
    std::ranges::to_input_view<DefaultInitializableView> to_input_view;
    assert(to_input_view.base().i_ == 94);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
