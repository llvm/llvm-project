//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <optional>
//
// template <class T>
//   constexpr optional<decay_t<T>> make_optional(T&& v);

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

template <class T, class = void>
constexpr bool can_make_optional_implicit = false;
template <class T>
constexpr bool can_make_optional_implicit<T, std::void_t<decltype(std::make_optional(std::declval<T>()))>> =
    mandate_same<decltype(std::make_optional(std::declval<T>())), std::optional<std::decay_t<T>>>::value;

static_assert(can_make_optional_implicit<int>);
static_assert(can_make_optional_implicit<const int>);
static_assert(can_make_optional_implicit<int&>);
static_assert(can_make_optional_implicit<const int&>);

static_assert(can_make_optional_implicit<std::unique_ptr<int>>);
static_assert(!can_make_optional_implicit<const std::unique_ptr<int>>);
static_assert(!can_make_optional_implicit<std::unique_ptr<int>&>);
static_assert(!can_make_optional_implicit<const std::unique_ptr<int>&>);

static_assert(can_make_optional_implicit<int[42]>);
static_assert(can_make_optional_implicit<const int (&)[42]>);

constexpr bool test() {
  {
    int arr[10]{};
    auto opt = std::make_optional(arr);
    ASSERT_SAME_TYPE(decltype(opt), std::optional<int*>);
    assert(*opt == arr);
  }
  {
    constexpr auto opt = std::make_optional(2);
    ASSERT_SAME_TYPE(decltype(opt), const std::optional<int>);
    static_assert(opt.value() == 2);
  }
  {
    auto opt = std::make_optional(2);
    ASSERT_SAME_TYPE(decltype(opt), std::optional<int>);
    assert(*opt == 2);
  }

  constexpr auto test_string = [] {
    const std::string s = "123";
    auto opt            = std::make_optional(s);
    ASSERT_SAME_TYPE(decltype(opt), std::optional<std::string>);
    assert(*opt == "123");
  };
  if (TEST_STD_AT_LEAST_20_OR_RUNTIME_EVALUATED)
    test_string();

  constexpr auto test_unique_ptr = [] {
    std::unique_ptr<int> s = std::make_unique<int>(3);
    auto opt               = std::make_optional(std::move(s));
    ASSERT_SAME_TYPE(decltype(opt), std::optional<std::unique_ptr<int>>);
    assert(**opt == 3);
    assert(s == nullptr);
  };
  if (TEST_STD_AT_LEAST_23_OR_RUNTIME_EVALUATED)
    test_unique_ptr();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
