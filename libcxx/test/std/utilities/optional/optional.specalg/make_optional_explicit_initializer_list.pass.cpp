//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// <optional>

// template <class T, class U, class... Args>
//   constexpr optional<T> make_optional(initializer_list<U> il, Args&&... args);

#include <cassert>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

#include "test_macros.h"

template <class T, class U>
struct mandate_same {
  ASSERT_SAME_TYPE(T, U);
  static constexpr bool value = true;
};

template <class V, class T, class... Us>
constexpr bool can_make_optional_explicit_impl = false;
template <class T, class... Us>
constexpr bool
    can_make_optional_explicit_impl<std::void_t<decltype(std::make_optional<T>(std::declval<Us>()...))>, T, Us...> =
        mandate_same<decltype(std::make_optional<T>(std::declval<Us>()...)), std::optional<T>>::value;

template <class T, class... Us>
constexpr bool can_make_optional_explicit = can_make_optional_explicit_impl<void, T, Us...>;

static_assert(!can_make_optional_explicit<int, std::initializer_list<int>&>);
static_assert(can_make_optional_explicit<std::string, std::initializer_list<char>&>);
static_assert(can_make_optional_explicit<std::string, std::initializer_list<char>&, std::allocator<char>>);
static_assert(!can_make_optional_explicit<std::string, std::initializer_list<std::string>&>);

#if TEST_STD_VER >= 26
static_assert(!can_make_optional_explicit<const int&, std::initializer_list<int>&>);
static_assert(can_make_optional_explicit<std::initializer_list<int>&, std::initializer_list<int>&>);
#endif

struct TestT {
  int x;
  int size;
  int *ptr;
  constexpr TestT(std::initializer_list<int> il)
    : x(*il.begin()), size(static_cast<int>(il.size())), ptr(nullptr) {}
  constexpr TestT(std::initializer_list<int> il, int *p)
    : x(*il.begin()), size(static_cast<int>(il.size())), ptr(p) {}
};

constexpr bool test() {
  {
    auto opt = std::make_optional<TestT>({42, 2, 3});
    ASSERT_SAME_TYPE(decltype(opt), std::optional<TestT>);
    assert(opt->x == 42);
    assert(opt->size == 3);
    assert(opt->ptr == nullptr);
  }
  {
    int i    = 42;
    auto opt = std::make_optional<TestT>({42, 2, 3}, &i);
    ASSERT_SAME_TYPE(decltype(opt), std::optional<TestT>);
    assert(opt->x == 42);
    assert(opt->size == 3);
    assert(opt->ptr == &i);
  }

  constexpr auto test_string = [] {
    {
      auto opt = std::make_optional<std::string>({'1', '2', '3'});
      assert(*opt == "123");
    }
    {
      auto opt = std::make_optional<std::string>({'a', 'b', 'c'}, std::allocator<char>{});
      assert(*opt == "abc");
    }
  };
  if (TEST_STD_AT_LEAST_20_OR_RUNTIME_EVALUATED)
    test_string();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
