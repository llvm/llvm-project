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

// template <class T> constexpr variant(T&&) noexcept(see below);

#include <cassert>
#include <string>
#include <type_traits>
#include <variant>

#include "test_convertible.hpp"
#include "test_macros.h"
#include "variant_test_helpers.hpp"

struct Dummy {
  Dummy() = default;
};

struct ThrowsT {
  ThrowsT(int) noexcept(false) {}
};

struct NoThrowT {
  NoThrowT(int) noexcept(true) {}
};

void test_T_ctor_noexcept() {
  {
    using V = std::variant<Dummy, NoThrowT>;
    static_assert(std::is_nothrow_constructible<V, int>::value, "");
  }
  {
    using V = std::variant<Dummy, ThrowsT>;
    static_assert(!std::is_nothrow_constructible<V, int>::value, "");
  }
}

void test_T_ctor_sfinae() {
  {
    using V = std::variant<long, unsigned>;
    static_assert(!std::is_constructible<V, int>::value, "ambiguous");
  }
  {
    using V = std::variant<std::string, std::string>;
    static_assert(!std::is_constructible<V, const char *>::value, "ambiguous");
  }
  {
    using V = std::variant<std::string, void *>;
    static_assert(!std::is_constructible<V, int>::value,
                  "no matching constructor");
  }
#if !defined(TEST_VARIANT_HAS_NO_REFERENCES)
  {
    using V = std::variant<int, int &&>;
    static_assert(!std::is_constructible<V, int>::value, "ambiguous");
  }
  {
    using V = std::variant<int, int const &>;
    static_assert(!std::is_constructible<V, int>::value, "ambiguous");
  }
#endif
}

void test_T_ctor_basic() {
  {
    constexpr std::variant<int> v(42);
    static_assert(v.index() == 0, "");
    static_assert(std::get<0>(v) == 42, "");
  }
  {
    constexpr std::variant<int, long> v(42l);
    static_assert(v.index() == 1, "");
    static_assert(std::get<1>(v) == 42, "");
  }
#if !defined(TEST_VARIANT_HAS_NO_REFERENCES)
  {
    using V = std::variant<int const &, int &&, long>;
    static_assert(std::is_convertible<int &, V>::value, "must be implicit");
    int x = 42;
    V v(x);
    assert(v.index() == 0);
    assert(&std::get<0>(v) == &x);
  }
  {
    using V = std::variant<int const &, int &&, long>;
    static_assert(std::is_convertible<int, V>::value, "must be implicit");
    int x = 42;
    V v(std::move(x));
    assert(v.index() == 1);
    assert(&std::get<1>(v) == &x);
  }
#endif
}

int main() {
  test_T_ctor_basic();
  test_T_ctor_noexcept();
  test_T_ctor_sfinae();
}
