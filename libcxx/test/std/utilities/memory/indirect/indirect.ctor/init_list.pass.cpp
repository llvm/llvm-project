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

// template<class I, class... Us>
//   constexpr explicit indirect(in_place_t, initializer_list<I> ilist, Us&&... us);

// template<class I, class... Us>
//   constexpr explicit indirect(allocator_arg_t, const Allocator& a,
//                               in_place_t, initializer_list<I> ilist, Us&&... us);

#include <cassert>
#include <concepts>
#include <initializer_list>
#include <type_traits>
#include <memory>

#include "test_convertible.h"
#include "test_allocator.h"
#include "min_allocator.h"

struct S {
  constexpr S(std::same_as<std::initializer_list<int>&> auto&& ilist_arg, std::same_as<int> auto&& i)
      : ilist(ilist_arg), int_addr(&i) {}

  std::initializer_list<int> ilist;
  int* int_addr;
};

constexpr void test_init_list_ctor_sfinae() {
  {
    using I = std::indirect<S>;
    static_assert(std::is_constructible_v<I, std::in_place_t, std::initializer_list<int>, int&&>);
    static_assert(std::is_constructible_v<I, std::in_place_t, const std::initializer_list<int>&, int&&>);
    static_assert(!std::is_constructible_v<I, std::in_place_t, std::initializer_list<int>&, int&>);
  }
  {
    using I = std::indirect<int, no_default_allocator<int>>;
    static_assert(std::is_constructible_v<I, std::allocator_arg_t, const no_default_allocator<int>&, int&>);
  }
}

constexpr void test_init_list_ctor_explicit() {
  static_assert(
      only_explicitly_constructible_from<std::indirect<int>, std::allocator_arg_t, const std::allocator<int>&, int&>);
  static_assert(
      only_explicitly_constructible_from<std::indirect<int>, std::allocator_arg_t, const std::allocator<int>&, int&&>);
}

constexpr void test_init_list_ctor() {
  {
    const std::initializer_list<int> ilist{1, 2};
    int n = 0;
    std::indirect<S> i(std::in_place, ilist, std::move(n));
    assert(!i.valueless_after_move());
    assert(i->ilist.begin() == ilist.begin());
    assert(i->ilist.end() == ilist.end());
    assert(i->int_addr == &n);
  }
  {
    const std::initializer_list<int> ilist{1, 2};
    int n = 0;
    std::indirect<S, test_allocator<S>> i(
        std::allocator_arg, test_allocator<S>(42), std::in_place, ilist, std::move(n));
    assert(!i.valueless_after_move());
    assert(i.get_allocator().get_data() == 42);
    assert(i->ilist.begin() == ilist.begin());
    assert(i->ilist.end() == ilist.end());
    assert(i->int_addr == &n);
  }
}

void test_init_list_ctor_throws() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  struct CtorThrows {
    CtorThrows(std::initializer_list<int>) { throw 42; }
  };

  try {
    std::indirect<CtorThrows> i(std::in_place, std::initializer_list<int>{});
    assert(false);
  } catch (const int& e) {
    assert(e == 42);
  } catch (...) {
    assert(false);
  }
#endif
}

constexpr bool test() {
  test_init_list_ctor_sfinae();
  test_init_list_ctor_explicit();
  test_init_list_ctor();

  return true;
}

int main(int, char**) {
  test_init_list_ctor_throws();
  test();
  static_assert(test());
  return 0;
}
