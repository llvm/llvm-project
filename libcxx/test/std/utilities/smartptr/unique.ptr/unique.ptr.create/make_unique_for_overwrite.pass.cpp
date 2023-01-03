//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<class T>
//   constexpr unique_ptr<T> make_unique_for_overwrite(); // T is not array
//
// template<class T>
//   constexpr unique_ptr<T> make_unique_for_overwrite(size_t n); // T is U[]
//
// template<class T, class... Args>
//   unspecified make_unique_for_overwrite(Args&&...) = delete; // T is U[N]

#include <cassert>
#include <concepts>
#include <cstring>
#include <memory>
#include <utility>

#include "test_macros.h"

template <class T, class... Args>
concept HasMakeUniqueForOverwrite =
    requires(Args&&... args) { std::make_unique_for_overwrite<T>(std::forward<Args>(args)...); };

struct Foo {
  int i;
};

// template<class T>
//   constexpr unique_ptr<T> make_unique_for_overwrite();
static_assert(HasMakeUniqueForOverwrite<int>);
static_assert(HasMakeUniqueForOverwrite<Foo>);
static_assert(!HasMakeUniqueForOverwrite<int, int>);
static_assert(!HasMakeUniqueForOverwrite<Foo, Foo>);

// template<class T>
//   constexpr unique_ptr<T> make_unique_for_overwrite(size_t n);
static_assert(HasMakeUniqueForOverwrite<int[], size_t>);
static_assert(HasMakeUniqueForOverwrite<Foo[], size_t>);
static_assert(!HasMakeUniqueForOverwrite<int[]>);
static_assert(!HasMakeUniqueForOverwrite<Foo[]>);
static_assert(!HasMakeUniqueForOverwrite<int[], size_t, int>);
static_assert(!HasMakeUniqueForOverwrite<Foo[], size_t, int>);

// template<class T, class... Args>
//   unspecified make_unique_for_overwrite(Args&&...) = delete;
static_assert(!HasMakeUniqueForOverwrite<int[2]>);
static_assert(!HasMakeUniqueForOverwrite<int[2], size_t>);
static_assert(!HasMakeUniqueForOverwrite<int[2], int>);
static_assert(!HasMakeUniqueForOverwrite<int[2], int, int>);
static_assert(!HasMakeUniqueForOverwrite<Foo[2]>);
static_assert(!HasMakeUniqueForOverwrite<Foo[2], size_t>);
static_assert(!HasMakeUniqueForOverwrite<Foo[2], int>);
static_assert(!HasMakeUniqueForOverwrite<Foo[2], int, int>);

struct WithDefaultConstructor {
  int i;
  constexpr WithDefaultConstructor() : i(5) {}
};

TEST_CONSTEXPR_CXX23 bool test() {
  // single int
  {
    std::same_as<std::unique_ptr<int>> decltype(auto) ptr = std::make_unique_for_overwrite<int>();
    // memory is available for write, otherwise constexpr test would fail
    *ptr = 5;
  }

  // unbounded array int[]
  {
    std::same_as<std::unique_ptr<int[]>> decltype(auto) ptrs = std::make_unique_for_overwrite<int[]>(3);

    // memory is available for write, otherwise constexpr test would fail
    ptrs[0] = 3;
    ptrs[1] = 4;
    ptrs[2] = 5;
  }

  // single with default constructor
  {
    std::same_as<std::unique_ptr<WithDefaultConstructor>> decltype(auto) ptr =
        std::make_unique_for_overwrite<WithDefaultConstructor>();
    assert(ptr->i == 5);
  }

  // unbounded array with default constructor
  {
    std::same_as<std::unique_ptr<WithDefaultConstructor[]>> decltype(auto) ptrs =
        std::make_unique_for_overwrite<WithDefaultConstructor[]>(3);
    assert(ptrs[0].i == 5);
    assert(ptrs[1].i == 5);
    assert(ptrs[2].i == 5);
  }

  return true;
}

// The standard specifically says to use `new (p) T`, which means that we should pick up any
// custom in-class operator new if there is one.
struct WithCustomNew {
  inline static bool customNewCalled    = false;
  inline static bool customNewArrCalled = false;

  static void* operator new(size_t n) {
    customNewCalled = true;
    return ::operator new(n);
    ;
  }

  static void* operator new[](size_t n) {
    customNewArrCalled = true;
    return ::operator new[](n);
  }
};

void testCustomNew() {
  // single with custom operator new
  {
    [[maybe_unused]] std::same_as<std::unique_ptr<WithCustomNew>> decltype(auto) ptr =
        std::make_unique_for_overwrite<WithCustomNew>();

    assert(WithCustomNew::customNewCalled);
  }

  // unbounded array with custom operator new
  {
    [[maybe_unused]] std::same_as<std::unique_ptr<WithCustomNew[]>> decltype(auto) ptr =
        std::make_unique_for_overwrite<WithCustomNew[]>(3);

    assert(WithCustomNew::customNewArrCalled);
  }
}

int main(int, char**) {
  test();
  testCustomNew();
#if TEST_STD_VER >= 23
  static_assert(test());
#endif

  return 0;
}
