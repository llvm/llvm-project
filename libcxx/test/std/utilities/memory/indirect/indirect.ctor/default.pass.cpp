//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <memory>

// template <class T, class Allocator = std::allocator<T>> class indirect;

// constexpr explicit indirect();

// constexpr explicit indirect(allocator_arg_t, const Allocator& a);

#include <cassert>
#include <type_traits>
#include <memory>

#include "min_allocator.h"
#include "test_allocator.h"
#include "test_convertible.h"

constexpr void test_default_ctor_sfinae() {
  static_assert(std::is_default_constructible_v<std::indirect<int>>);
  static_assert(!std::is_default_constructible_v<std::indirect<int, no_default_allocator<int>>>);
}

constexpr void test_default_ctor_explicit() {
  static_assert(only_explicitly_constructible_from<std::indirect<int>>);
  static_assert(
      only_explicitly_constructible_from<std::indirect<int>, std::allocator_arg_t, const std::allocator<int>&>);
}

constexpr void test_default_ctor() {
  {
    std::indirect<int> i;
    assert(!i.valueless_after_move());
    assert(*i == 0);
  }
  {
    std::indirect<int, test_allocator<int>> i(std::allocator_arg, test_allocator<int>(42));
    assert(!i.valueless_after_move());
    assert(*i == 0);
    assert(i.get_allocator().get_data() == 42);
  }
}

void test_default_ctor_throws() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  struct DefaultCtorThrows {
    DefaultCtorThrows() { throw 42; }
  };

  try {
    std::indirect<DefaultCtorThrows> i;
    assert(false);
  } catch (const int& e) {
    assert(e == 42);
  } catch (...) {
    assert(false);
  }

  try {
    std::indirect<DefaultCtorThrows> i(std::allocator_arg, std::allocator<int>());
    assert(false);
  } catch (const int& e) {
    assert(e == 42);
  } catch (...) {
    assert(false);
  }
#endif
}

constexpr bool test() {
  test_default_ctor_sfinae();
  test_default_ctor_explicit();
  test_default_ctor();

  return true;
}

int main(int, char**) {
  test_default_ctor_throws();
  test();
  static_assert(test());
  return 0;
}
