// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// Throwing bad_variant_access is supported starting in macosx10.13
// XFAIL: with_system_cxx_lib=macosx10.12 && !no-exceptions
// XFAIL: with_system_cxx_lib=macosx10.11 && !no-exceptions
// XFAIL: with_system_cxx_lib=macosx10.10 && !no-exceptions
// XFAIL: with_system_cxx_lib=macosx10.9 && !no-exceptions

// <variant>

// template <class ...Types> class variant;

// constexpr variant() noexcept(see below);

#include <cassert>
#include <type_traits>
#include <variant>

#include "test_macros.h"
#include "variant_test_helpers.h"

struct NonDefaultConstructible {
  constexpr NonDefaultConstructible(int) {}
};

struct NotNoexcept {
  NotNoexcept() noexcept(false) {}
};

#ifndef TEST_HAS_NO_EXCEPTIONS
struct DefaultCtorThrows {
  DefaultCtorThrows() { throw 42; }
};
#endif

void test_default_ctor_sfinae() {
  {
    using V = std::variant<std::monostate, int>;
    static_assert(std::is_default_constructible<V>::value, "");
  }
  {
    using V = std::variant<NonDefaultConstructible, int>;
    static_assert(!std::is_default_constructible<V>::value, "");
  }
#if !defined(TEST_VARIANT_HAS_NO_REFERENCES)
  {
    using V = std::variant<int &, int>;
    static_assert(!std::is_default_constructible<V>::value, "");
  }
#endif
}

void test_default_ctor_noexcept() {
  {
    using V = std::variant<int>;
    static_assert(std::is_nothrow_default_constructible<V>::value, "");
  }
  {
    using V = std::variant<NotNoexcept>;
    static_assert(!std::is_nothrow_default_constructible<V>::value, "");
  }
}

void test_default_ctor_throws() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  using V = std::variant<DefaultCtorThrows, int>;
  try {
    V v;
    assert(false);
  } catch (const int &ex) {
    assert(ex == 42);
  } catch (...) {
    assert(false);
  }
#endif
}

void test_default_ctor_basic() {
  {
    std::variant<int> v;
    assert(v.index() == 0);
    assert(std::get<0>(v) == 0);
  }
  {
    std::variant<int, long> v;
    assert(v.index() == 0);
    assert(std::get<0>(v) == 0);
  }
  {
    std::variant<int, NonDefaultConstructible> v;
    assert(v.index() == 0);
    assert(std::get<0>(v) == 0);
  }
  {
    using V = std::variant<int, long>;
    constexpr V v;
    static_assert(v.index() == 0, "");
    static_assert(std::get<0>(v) == 0, "");
  }
  {
    using V = std::variant<int, long>;
    constexpr V v;
    static_assert(v.index() == 0, "");
    static_assert(std::get<0>(v) == 0, "");
  }
  {
    using V = std::variant<int, NonDefaultConstructible>;
    constexpr V v;
    static_assert(v.index() == 0, "");
    static_assert(std::get<0>(v) == 0, "");
  }
}

int main(int, char**) {
  test_default_ctor_basic();
  test_default_ctor_sfinae();
  test_default_ctor_noexcept();
  test_default_ctor_throws();

  return 0;
}
