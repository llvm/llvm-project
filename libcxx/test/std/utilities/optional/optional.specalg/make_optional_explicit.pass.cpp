//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// <optional>

// template <class T, class... Args>
//   constexpr optional<T> make_optional(Args&&... args);

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

static_assert(can_make_optional_explicit<int, long>);
static_assert(can_make_optional_explicit<int, const long&>);
static_assert(can_make_optional_explicit<std::string, const char*>);
static_assert(can_make_optional_explicit<std::string, const char* const&>);
static_assert(can_make_optional_explicit<std::unique_ptr<const int>, std::unique_ptr<int>>);
static_assert(!can_make_optional_explicit<std::unique_ptr<const int>, const std::unique_ptr<int>>);
static_assert(!can_make_optional_explicit<std::unique_ptr<const int>, std::unique_ptr<int>&>);
static_assert(!can_make_optional_explicit<std::unique_ptr<const int>, const std::unique_ptr<int>&>);

#if TEST_STD_VER >= 26
static_assert(can_make_optional_explicit<const int&, int&>);
static_assert(can_make_optional_explicit<const int&, const int&>);
static_assert(!can_make_optional_explicit<const int&, int>);
static_assert(!can_make_optional_explicit<const int&, const int>);
static_assert(!can_make_optional_explicit<const int&, long&>);
static_assert(!can_make_optional_explicit<const int&, const long&>);
static_assert(!can_make_optional_explicit<const int&, long>);
static_assert(!can_make_optional_explicit<const int&, const long&>);
#endif

template <typename T>
constexpr void test_ref() {
  T i{0};
  auto opt = std::make_optional<T&>(i);

#if TEST_STD_VER < 26
  ASSERT_SAME_TYPE(decltype(opt), std::optional<T>);
#else
  ASSERT_SAME_TYPE(decltype(opt), std::optional<T&>);
#endif

  assert(*opt == 0);
}

constexpr bool test() {
  {
    constexpr auto opt = std::make_optional<int>('a');
    static_assert(*opt == int('a'));
  }

  constexpr auto test_string = [] {
    {
      std::string s = "123";
      auto opt      = std::make_optional<std::string>(s);
      assert(*opt == "123");
    }
    {
      auto opt = std::make_optional<std::string>(4u, 'X');
      assert(*opt == "XXXX");
    }
  };
  if (TEST_STD_AT_LEAST_20_OR_RUNTIME_EVALUATED)
    test_string();

  constexpr auto test_unique_ptr = [] {
    std::unique_ptr<int> s = std::make_unique<int>(3);
    auto opt               = std::make_optional<std::unique_ptr<int>>(std::move(s));
    assert(**opt == 3);
    assert(s == nullptr);
  };
  if (TEST_STD_AT_LEAST_23_OR_RUNTIME_EVALUATED)
    test_unique_ptr();

  test_ref<int>();
  test_ref<double>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
