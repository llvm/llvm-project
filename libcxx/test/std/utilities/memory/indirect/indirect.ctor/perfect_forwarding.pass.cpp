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

// template<class U = T>
//  constexpr explicit indirect(U&& u);

// template<class U = T>
//   constexpr explicit indirect(allocator_arg_t, const Allocator& a, U&& u);

#include <cassert>
#include <type_traits>
#include <memory>

#include "test_convertible.h"
#include "min_allocator.h"
#include "archetypes.h"

constexpr void test_perfect_forwarding_ctor_sfinae() {
  {
    using I = std::indirect<TestTypes::MoveOnly>;
    static_assert(!std::is_constructible_v<I, TestTypes::MoveOnly&>);
    static_assert(!std::is_constructible_v<I, const TestTypes::MoveOnly&>);
    static_assert(std::is_constructible_v<I, TestTypes::MoveOnly&&>);
  }
  { // If the allocator isn't default-constructible, only the uses-allocator constructor is enabled.
    using I = std::indirect<int, no_default_allocator<int>>;
    static_assert(!std::is_constructible_v<I, int&>);
    static_assert(std::is_constructible_v<I, std::allocator_arg_t, const no_default_allocator<int>&, int&>);
  }
}

constexpr void test_perfect_forwarding_ctor_explicit() {
  static_assert(only_explicitly_constructible_from<std::indirect<int>, int&>);
  static_assert(only_explicitly_constructible_from<std::indirect<int, no_default_allocator<int>>,
                                                   std::allocator_arg_t,
                                                   const no_default_allocator<int>&,
                                                   int&>);
}

constexpr void test_perfect_forwarding_ctor() {
  {
    std::indirect<int> i(42);
    assert(!i.valueless_after_move());
    assert(*i == 42);
  }
  {
    std::indirect<int> i(std::allocator_arg, std::allocator<int>(), 42);
    assert(!i.valueless_after_move());
    assert(*i == 42);
  }
}

void test_perfect_forwarding_ctor_throws() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  struct CopyCtorThrows {
    CopyCtorThrows() = default;
    CopyCtorThrows(const CopyCtorThrows&) { throw 42; }
  };

  try {
    std::indirect<CopyCtorThrows> i(CopyCtorThrows{});
    assert(false);
  } catch (const int& e) {
    assert(e == 42);
  } catch (...) {
    assert(false);
  }

  try {
    std::indirect<CopyCtorThrows> i(std::allocator_arg, std::allocator<CopyCtorThrows>(), CopyCtorThrows{});
    assert(false);
  } catch (const int& e) {
    assert(e == 42);
  } catch (...) {
    assert(false);
  }
#endif
}

constexpr bool test() {
  test_perfect_forwarding_ctor_sfinae();
  test_perfect_forwarding_ctor_explicit();
  test_perfect_forwarding_ctor();

  return true;
}

int main(int, char**) {
  test_perfect_forwarding_ctor_throws();
  test();
  static_assert(test());
  return 0;
}
