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

// template<class... Us>
//   constexpr explicit indirect(in_place_t, Us&&... us);

// template<class... Us>
//   constexpr explicit indirect(allocator_arg_t, const Allocator& a,
//                               in_place_t, Us&& ...us);

#include <cassert>
#include <type_traits>
#include <memory>

#include "test_convertible.h"
#include "test_allocator.h"
#include "min_allocator.h"
#include "archetypes.h"

constexpr void test_in_place_t_ctor_sfinae() {
  {
    static_assert(
        !std::is_constructible_v<std::indirect<TestTypes::MoveOnly>, std::in_place_t, const TestTypes::MoveOnly&>);
    static_assert(std::is_constructible_v<std::indirect<TestTypes::MoveOnly>, std::in_place_t, TestTypes::MoveOnly&&>);
  }
  {
    static_assert(std::is_constructible_v<std::indirect<int, no_default_allocator<int>>,
                                          std::allocator_arg_t,
                                          const no_default_allocator<int>&,
                                          std::in_place_t,
                                          int&>);
  }
  {
    static_assert(!std::is_constructible_v<std::indirect<TestTypes::MoveOnly>,
                                           std::allocator_arg_t,
                                           const std::allocator<TestTypes::MoveOnly>&,
                                           std::in_place_t,
                                           const TestTypes::MoveOnly&>);
    static_assert(std::is_constructible_v<std::indirect<TestTypes::MoveOnly>,
                                          std::allocator_arg_t,
                                          const std::allocator<TestTypes::MoveOnly>&,
                                          std::in_place_t,
                                          TestTypes::MoveOnly&&>);
  }
}

constexpr void test_in_place_t_ctor_explicit() {
  static_assert(only_explicitly_constructible_from<std::indirect<int>, std::in_place_t, int&>);
  static_assert(only_explicitly_constructible_from<std::indirect<int>,
                                                   std::allocator_arg_t,
                                                   const std::allocator<TestTypes::MoveOnly>&,
                                                   std::in_place_t,
                                                   int&>);
}

constexpr void test_in_place_t_ctor() {
  {
    std::indirect<int> i(std::in_place, 42);
    assert(!i.valueless_after_move());
    assert(*i == 42);
  }
  {
    std::indirect<std::pair<int, int>> i(std::in_place, 1, 2);
    assert(!i.valueless_after_move());
    assert((*i == std::pair{1, 2}));
  }
  {
    std::indirect<int, test_allocator<int>> i(std::allocator_arg, test_allocator<int>(67), std::in_place, 42);
    assert(!i.valueless_after_move());
    assert(i.get_allocator().get_data() == 67);
    assert(*i == 42);
  }
  {
    std::indirect<std::pair<int, int>, test_allocator<std::pair<int, int>>> i(
        std::allocator_arg, test_allocator<int>(67), std::in_place, 1, 2);
    assert(!i.valueless_after_move());
    assert(i.get_allocator().get_data() == 67);
    assert((*i == std::pair{1, 2}));
  }
}

void test_in_place_t_ctor_throws() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  struct CopyCtorThrows {
    CopyCtorThrows() = default;
    CopyCtorThrows(const CopyCtorThrows&) { throw 42; }
  };

  CopyCtorThrows c;
  try {
    std::indirect<CopyCtorThrows> i(std::in_place, c);
    assert(false);
  } catch (const int& e) {
    assert(e == 42);
  } catch (...) {
    assert(false);
  }
#endif
}

constexpr bool test() {
  test_in_place_t_ctor_sfinae();
  test_in_place_t_ctor_explicit();
  test_in_place_t_ctor();

  return true;
}

int main(int, char**) {
  test_in_place_t_ctor_throws();
  test();
  static_assert(test());
  return 0;
}
