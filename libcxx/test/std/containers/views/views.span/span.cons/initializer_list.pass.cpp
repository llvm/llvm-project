//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <span>

// constexpr explicit(extent != dynamic_extent) span(std::initializer_list<value_type> il); // Since C++26

#include <any>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <span>
#include <type_traits>

#include "test_convertible.h"
#include "test_macros.h"

#if TEST_STD_VER >= 26

// SFINAE

template <typename T>
concept ConstElementType = std::is_const_v<typename T::element_type>;

static_assert(ConstElementType<std::span<const int>>);
static_assert(!ConstElementType<std::span<int>>);
static_assert(ConstElementType<std::span<const int, 94>>);
static_assert(!ConstElementType<std::span<int, 94>>);

// Constructor constraings

template <typename I, typename T, std::size_t... N>
concept HasInitializerListCtr = requires(I il) { std::span<T, N...>{il}; };

static_assert(HasInitializerListCtr<std::initializer_list<const int>, const int>);
static_assert(!HasInitializerListCtr<std::initializer_list<int>, int>);
static_assert(HasInitializerListCtr<std::initializer_list<const int>, const int, 94>);
static_assert(!HasInitializerListCtr<std::initializer_list<int>, int, 94>);

// Constructor conditionally explicit

static_assert(!test_convertible<std::span<const int, 28>, std::initializer_list<int>>(),
              "This constructor must be explicit");
static_assert(std::is_constructible_v<std::span<const int, 28>, std::initializer_list<int>>);
static_assert(test_convertible<std::span<const int>, std::initializer_list<int>>(),
              "This constructor must not be explicit");
static_assert(std::is_constructible_v<std::span<const int>, std::initializer_list<int>>);

#endif

struct Sink {
  constexpr Sink() = default;
  constexpr Sink(Sink*) {}
};

constexpr std::size_t count(std::span<const Sink> sp) { return sp.size(); }

template <std::size_t N>
constexpr std::size_t count_n(std::span<const Sink, N> sp) {
  return sp.size();
}

constexpr bool test() {
#if TEST_STD_VER >= 26
  // Dynamic extent
  {
    Sink a[10];

    assert(count({a}) == 1);
    assert(count({a, a + 10}) == 2);
    assert(count({a, a + 1, a + 2}) == 3);
    assert(count(std::initializer_list<Sink>{a[0], a[1], a[2], a[3]}) == 4);
  }
#else
  {
    Sink a[10];

    assert(count({a}) == 10);
    assert(count({a, a + 10}) == 10);
    assert(count_n<10>({a}) == 10);
  }
#endif

  return true;
}

// Test P2447R4 "Annex C examples"

constexpr int three(std::span<void* const> sp) { return sp.size(); }

constexpr int four(std::span<const std::any> sp) { return sp.size(); }

bool test_P2447R4_annex_c_examples() {
  // 1. Overload resolution is affected
  // --> tested in "initializer_list.verify.cpp"

  // 2. The `initializer_list` ctor has high precedence
  // --> tested in "initializer_list.verify.cpp"

  // 3. Implicit two-argument construction with a highly convertible value_type
#if TEST_STD_VER >= 26
  {
    void* a[10];
    assert(three({a, 0}) == 2);
  }
  {
    std::any a[10];
    assert(four({a, a + 10}) == 2);
  }
#else
  {
    void* a[10];
    assert(three({a, 0}) == 0);
  }
  {
    std::any a[10];
    assert(four({a, a + 10}) == 10);
  }
#endif

  return true;
}

int main(int, char**) {
  assert(test());
  static_assert(test());

  assert(test_P2447R4_annex_c_examples());

  return 0;
}
