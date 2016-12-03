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

// template <size_t I, class ...Args> void emplace(Args&&... args);

#include <cassert>
#include <string>
#include <type_traits>
#include <variant>

#include "archetypes.hpp"
#include "test_convertible.hpp"
#include "test_macros.h"
#include "variant_test_helpers.hpp"

template <class Var, size_t I, class... Args>
constexpr auto test_emplace_exists_imp(int) -> decltype(
    std::declval<Var>().template emplace<I>(std::declval<Args>()...), true) {
  return true;
}

template <class, size_t, class...>
constexpr auto test_emplace_exists_imp(long) -> bool {
  return false;
}

template <class Var, size_t I, class... Args> constexpr bool emplace_exists() {
  return test_emplace_exists_imp<Var, I, Args...>(0);
}

void test_emplace_sfinae() {
  {
    using V = std::variant<int, void *, const void *, TestTypes::NoCtors>;
    static_assert(emplace_exists<V, 0>(), "");
    static_assert(emplace_exists<V, 0, int>(), "");
    static_assert(!emplace_exists<V, 0, decltype(nullptr)>(),
                  "cannot construct");
    static_assert(emplace_exists<V, 1, decltype(nullptr)>(), "");
    static_assert(emplace_exists<V, 1, int *>(), "");
    static_assert(!emplace_exists<V, 1, const int *>(), "");
    static_assert(!emplace_exists<V, 1, int>(), "cannot construct");
    static_assert(emplace_exists<V, 2, const int *>(), "");
    static_assert(emplace_exists<V, 2, int *>(), "");
    static_assert(!emplace_exists<V, 3>(), "cannot construct");
  }
#if !defined(TEST_VARIANT_HAS_NO_REFERENCES)
  {
    using V = std::variant<int, int &, const int &, int &&, TestTypes::NoCtors>;
    static_assert(emplace_exists<V, 0>(), "");
    static_assert(emplace_exists<V, 0, int>(), "");
    static_assert(emplace_exists<V, 0, long long>(), "");
    static_assert(!emplace_exists<V, 0, int, int>(), "too many args");
    static_assert(emplace_exists<V, 1, int &>(), "");
    static_assert(!emplace_exists<V, 1>(), "cannot default construct ref");
    static_assert(!emplace_exists<V, 1, const int &>(), "cannot bind ref");
    static_assert(!emplace_exists<V, 1, int &&>(), "cannot bind ref");
    static_assert(emplace_exists<V, 2, int &>(), "");
    static_assert(emplace_exists<V, 2, const int &>(), "");
    static_assert(emplace_exists<V, 2, int &&>(), "");
    static_assert(!emplace_exists<V, 2, void *>(),
                  "not constructible from void*");
    static_assert(emplace_exists<V, 3, int>(), "");
    static_assert(!emplace_exists<V, 3, int &>(), "cannot bind ref");
    static_assert(!emplace_exists<V, 3, const int &>(), "cannot bind ref");
    static_assert(!emplace_exists<V, 3, const int &&>(), "cannot bind ref");
    static_assert(!emplace_exists<V, 4>(), "no ctors");
  }
#endif
}

void test_basic() {
  {
    using V = std::variant<int>;
    V v(42);
    v.emplace<0>();
    assert(std::get<0>(v) == 0);
    v.emplace<0>(42);
    assert(std::get<0>(v) == 42);
  }
  {
    using V =
        std::variant<int, long, const void *, TestTypes::NoCtors, std::string>;
    const int x = 100;
    V v(std::in_place_index<0>, -1);
    // default emplace a value
    v.emplace<1>();
    assert(std::get<1>(v) == 0);
    v.emplace<2>(&x);
    assert(std::get<2>(v) == &x);
    // emplace with multiple args
    v.emplace<4>(3, 'a');
    assert(std::get<4>(v) == "aaa");
  }
#if !defined(TEST_VARIANT_HAS_NO_REFERENCES)
  {
    using V = std::variant<int, long, const int &, int &&, TestTypes::NoCtors,
                           std::string>;
    const int x = 100;
    int y = 42;
    int z = 43;
    V v(std::in_place_index<0>, -1);
    // default emplace a value
    v.emplace<1>();
    assert(std::get<1>(v) == 0);
    // emplace a reference
    v.emplace<2>(x);
    assert(&std::get<2>(v) == &x);
    // emplace an rvalue reference
    v.emplace<3>(std::move(y));
    assert(&std::get<3>(v) == &y);
    // re-emplace a new reference over the active member
    v.emplace<3>(std::move(z));
    assert(&std::get<3>(v) == &z);
    // emplace with multiple args
    v.emplace<5>(3, 'a');
    assert(std::get<5>(v) == "aaa");
  }
#endif
}

int main() {
  test_basic();
  test_emplace_sfinae();
}
