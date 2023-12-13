//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<class U, class... Args>
//   constexpr explicit expected(unexpect_t, initializer_list<U> il, Args&&... args);
//
// Constraints: is_constructible_v<E, initializer_list<U>&, Args...> is true.
//
// Effects: Direct-non-list-initializes unex with il, std::forward<Args>(args)....
//
// Postconditions: has_value() is false.
//
// Throws: Any exception thrown by the initialization of unex.

#include <algorithm>
#include <cassert>
#include <expected>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "MoveOnly.h"
#include "test_macros.h"
#include "../../types.h"

// Test Constraints:
static_assert(
    std::is_constructible_v<std::expected<int, std::vector<int>>, std::unexpect_t, std::initializer_list<int>>);

// !is_constructible_v<T, initializer_list<U>&, Args...>
static_assert(!std::is_constructible_v<std::expected<int, int>, std::unexpect_t, std::initializer_list<int>>);

// test explicit
template <class T>
void conversion_test(T);

template <class T, class... Args>
concept ImplicitlyConstructible = requires(Args&&... args) { conversion_test<T>({std::forward<Args>(args)...}); };
static_assert(ImplicitlyConstructible<int, int>);

static_assert(
    !ImplicitlyConstructible<std::expected<int, std::vector<int>>, std::unexpect_t, std::initializer_list<int>>);

template <class... Ts>
struct Data {
  std::vector<int> vec_;
  std::tuple<Ts...> tuple_;

  template <class... Us>
    requires std::is_constructible_v<std::tuple<Ts...>, Us&&...>
  constexpr Data(std::initializer_list<int> il, Us&&... us) : vec_(il), tuple_(std::forward<Us>(us)...) {}
};

constexpr bool test() {
  // no arg
  {
    std::expected<int, Data<>> e(std::unexpect, {1, 2, 3});
    assert(!e.has_value());
    auto expectedList = {1, 2, 3};
    assert(std::ranges::equal(e.error().vec_, expectedList));
  }

  // one arg
  {
    std::expected<int, Data<MoveOnly>> e(std::unexpect, {4, 5, 6}, MoveOnly(5));
    assert(!e.has_value());
    auto expectedList = {4, 5, 6};
    assert((std::ranges::equal(e.error().vec_, expectedList)));
    assert(std::get<0>(e.error().tuple_) == 5);
  }

  // multi args
  {
    int i = 5;
    int j = 6;
    MoveOnly m(7);
    std::expected<int, Data<int&, int&&, MoveOnly>> e(std::unexpect, {1, 2}, i, std::move(j), std::move(m));
    assert(!e.has_value());
    auto expectedList = {1, 2};
    assert((std::ranges::equal(e.error().vec_, expectedList)));
    assert(&std::get<0>(e.error().tuple_) == &i);
    assert(&std::get<1>(e.error().tuple_) == &j);
    assert(std::get<2>(e.error().tuple_) == 7);
    assert(m.get() == 0);
  }

  // TailClobberer
  {
    std::expected<bool, TailClobberer<1>> e(std::unexpect, {1, 2, 3});
    assert(!e.has_value());
  }

  return true;
}

void testException() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  struct Throwing {
    Throwing(std::initializer_list<int>, int) { throw Except{}; };
  };

  try {
    std::expected<int, Throwing> u(std::unexpect, {1, 2}, 5);
    assert(false);
  } catch (Except) {
  }
#endif // TEST_HAS_NO_EXCEPTIONS
}

int main(int, char**) {
  test();
  static_assert(test());
  testException();
  return 0;
}
