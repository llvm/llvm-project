//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <variant>

// template <class ...Types> class variant;

// template <class T, class ...Args> T& emplace(Args&&... args);

#include <cassert>
#include <string>
#include <type_traits>
#include <variant>

#include "archetypes.h"
#include "test_convertible.h"
#include "test_macros.h"
#include "variant_test_helpers.h"

template <class Var, class T, class... Args>
constexpr auto test_emplace_exists_imp(int)
    -> decltype(std::declval<Var>().template emplace<T>(std::declval<Args>()...), true) {
  return true;
}

template <class, class, class...>
constexpr auto test_emplace_exists_imp(long) -> bool {
  return false;
}

template <class... Args>
constexpr bool emplace_exists() {
  return test_emplace_exists_imp<Args...>(0);
}

constexpr void test_emplace_sfinae() {
  {
    using V = std::variant<int, void*, const void*, TestTypes::NoCtors>;
    static_assert(emplace_exists<V, int>(), "");
    static_assert(emplace_exists<V, int, int>(), "");
    static_assert(!emplace_exists<V, int, decltype(nullptr)>(), "cannot construct");
    static_assert(emplace_exists<V, void*, decltype(nullptr)>(), "");
    static_assert(!emplace_exists<V, void*, int>(), "cannot construct");
    static_assert(emplace_exists<V, void*, int*>(), "");
    static_assert(!emplace_exists<V, void*, const int*>(), "");
    static_assert(emplace_exists<V, const void*, const int*>(), "");
    static_assert(emplace_exists<V, const void*, int*>(), "");
    static_assert(!emplace_exists<V, TestTypes::NoCtors>(), "cannot construct");
  }
}

struct NoCtor {
  NoCtor() = delete;
};

TEST_CONSTEXPR_CXX20 void test_basic() {
  {
    using V = std::variant<int>;
    V v(42);
    auto& ref1 = v.emplace<int>();
    static_assert(std::is_same_v<int&, decltype(ref1)>, "");
    assert(std::get<0>(v) == 0);
    assert(&ref1 == &std::get<0>(v));
    auto& ref2 = v.emplace<int>(42);
    static_assert(std::is_same_v<int&, decltype(ref2)>, "");
    assert(std::get<0>(v) == 42);
    assert(&ref2 == &std::get<0>(v));
  }
  {
    using V     = std::variant<int, long, const void*, NoCtor, std::string>;
    const int x = 100;
    V v(std::in_place_type<int>, -1);
    // default emplace a value
    auto& ref1 = v.emplace<long>();
    static_assert(std::is_same_v<long&, decltype(ref1)>, "");
    assert(std::get<1>(v) == 0);
    assert(&ref1 == &std::get<1>(v));
    auto& ref2 = v.emplace<const void*>(&x);
    static_assert(std::is_same_v<const void*&, decltype(ref2)>, "");
    assert(std::get<2>(v) == &x);
    assert(&ref2 == &std::get<2>(v));
    // emplace with multiple args
    auto& ref3 = v.emplace<std::string>(3u, 'a');
    static_assert(std::is_same_v<std::string&, decltype(ref3)>, "");
    assert(std::get<4>(v) == "aaa");
    assert(&ref3 == &std::get<4>(v));
  }
}

TEST_CONSTEXPR_CXX20 bool test() {
  test_basic();
  test_emplace_sfinae();

  return true;
}

int main(int, char**) {
  test();

#if TEST_STD_VER >= 20
  static_assert(test());
#endif

  return 0;
}
