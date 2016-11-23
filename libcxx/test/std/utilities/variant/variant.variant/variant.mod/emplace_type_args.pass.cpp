// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14

// <variant>

// template <class ...Types> class variant;

// template <class T, class ...Args> void emplace(Args&&... args);

#include <cassert>
#include <string>
#include <type_traits>
#include <variant>

#include "archetypes.hpp"
#include "test_convertible.hpp"
#include "test_macros.h"
#include "variant_test_helpers.hpp"

template <class Var, class T, class... Args>
constexpr auto test_emplace_exists_imp(int) -> decltype(
    std::declval<Var>().template emplace<T>(std::declval<Args>()...), true) {
  return true;
}

template <class, class, class...>
constexpr auto test_emplace_exists_imp(long) -> bool {
  return false;
}

template <class... Args> constexpr bool emplace_exists() {
  return test_emplace_exists_imp<Args...>(0);
}

void test_emplace_sfinae() {
  {
    using V = std::variant<int, void *, const void *, TestTypes::NoCtors>;
    static_assert(emplace_exists<V, int>(), "");
    static_assert(emplace_exists<V, int, int>(), "");
    static_assert(!emplace_exists<V, int, decltype(nullptr)>(),
                  "cannot construct");
    static_assert(emplace_exists<V, void *, decltype(nullptr)>(), "");
    static_assert(!emplace_exists<V, void *, int>(), "cannot construct");
    static_assert(emplace_exists<V, void *, int *>(), "");
    static_assert(!emplace_exists<V, void *, const int *>(), "");
    static_assert(emplace_exists<V, void const *, const int *>(), "");
    static_assert(emplace_exists<V, void const *, int *>(), "");
    static_assert(!emplace_exists<V, TestTypes::NoCtors>(), "cannot construct");
  }
#if !defined(TEST_VARIANT_HAS_NO_REFERENCES)
  using V = std::variant<int, int &, int const &, int &&, long, long,
                         TestTypes::NoCtors>;
  static_assert(emplace_exists<V, int>(), "");
  static_assert(emplace_exists<V, int, int>(), "");
  static_assert(emplace_exists<V, int, long long>(), "");
  static_assert(!emplace_exists<V, int, int, int>(), "too many args");
  static_assert(emplace_exists<V, int &, int &>(), "");
  static_assert(!emplace_exists<V, int &>(), "cannot default construct ref");
  static_assert(!emplace_exists<V, int &, int const &>(), "cannot bind ref");
  static_assert(!emplace_exists<V, int &, int &&>(), "cannot bind ref");
  static_assert(emplace_exists<V, int const &, int &>(), "");
  static_assert(emplace_exists<V, int const &, const int &>(), "");
  static_assert(emplace_exists<V, int const &, int &&>(), "");
  static_assert(!emplace_exists<V, int const &, void *>(),
                "not constructible from void*");
  static_assert(emplace_exists<V, int &&, int>(), "");
  static_assert(!emplace_exists<V, int &&, int &>(), "cannot bind ref");
  static_assert(!emplace_exists<V, int &&, int const &>(), "cannot bind ref");
  static_assert(!emplace_exists<V, int &&, int const &&>(), "cannot bind ref");
  static_assert(!emplace_exists<V, long, long>(), "ambiguous");
  static_assert(!emplace_exists<V, TestTypes::NoCtors>(),
                "cannot construct void");
#endif
}

void test_basic() {
  {
    using V = std::variant<int>;
    V v(42);
    v.emplace<int>();
    assert(std::get<0>(v) == 0);
    v.emplace<int>(42);
    assert(std::get<0>(v) == 42);
  }
  {
    using V =
        std::variant<int, long, const void *, TestTypes::NoCtors, std::string>;
    const int x = 100;
    V v(std::in_place_type<int>, -1);
    // default emplace a value
    v.emplace<long>();
    assert(std::get<1>(v) == 0);
    v.emplace<const void *>(&x);
    assert(std::get<2>(v) == &x);
    // emplace with multiple args
    v.emplace<std::string>(3, 'a');
    assert(std::get<4>(v) == "aaa");
  }
#if !defined(TEST_VARIANT_HAS_NO_REFERENCES)
  {
    using V = std::variant<int, long, int const &, int &&, TestTypes::NoCtors,
                           std::string>;
    const int x = 100;
    int y = 42;
    int z = 43;
    V v(std::in_place_index<0>, -1);
    // default emplace a value
    v.emplace<long>();
    assert(std::get<long>(v) == 0);
    // emplace a reference
    v.emplace<int const &>(x);
    assert(&std::get<int const &>(v) == &x);
    // emplace an rvalue reference
    v.emplace<int &&>(std::move(y));
    assert(&std::get<int &&>(v) == &y);
    // re-emplace a new reference over the active member
    v.emplace<int &&>(std::move(z));
    assert(&std::get<int &&>(v) == &z);
    // emplace with multiple args
    v.emplace<std::string>(3, 'a');
    assert(std::get<std::string>(v) == "aaa");
  }
#endif
}

int main() {
  test_basic();
  test_emplace_sfinae();
}
