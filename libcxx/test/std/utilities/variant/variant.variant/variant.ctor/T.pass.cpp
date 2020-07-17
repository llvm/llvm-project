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

// template <class T> constexpr variant(T&&) noexcept(see below);

#include <cassert>
#include <string>
#include <type_traits>
#include <variant>
#include <memory>

#include "test_macros.h"
#include "variant_test_helpers.h"

struct Dummy {
  Dummy() = default;
};

struct ThrowsT {
  ThrowsT(int) noexcept(false) {}
};

struct NoThrowT {
  NoThrowT(int) noexcept(true) {}
};

struct AnyConstructible { template <typename T> AnyConstructible(T&&) {} };
struct NoConstructible { NoConstructible() = delete; };
template <class T>
struct RValueConvertibleFrom { RValueConvertibleFrom(T&&) {} };

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
    using V = std::variant<long, long long>;
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
  {
    using V = std::variant<std::string, float>;
    static_assert(std::is_constructible<V, int>::value == VariantAllowsNarrowingConversions,
                  "no matching constructor");
  }
  {
    using V = std::variant<std::unique_ptr<int>, bool>;
    static_assert(!std::is_constructible<V, std::unique_ptr<char>>::value,
                  "no explicit bool in constructor");
    struct X {
      operator void*();
    };
    static_assert(!std::is_constructible<V, X>::value,
                  "no boolean conversion in constructor");
    static_assert(!std::is_constructible<V, std::false_type>::value,
                  "no converted to bool in constructor");
  }
  {
    struct X {};
    struct Y {
      operator X();
    };
    using V = std::variant<X>;
    static_assert(std::is_constructible<V, Y>::value,
                  "regression on user-defined conversions in constructor");
  }
  {
    using V = std::variant<AnyConstructible, NoConstructible>;
    static_assert(
        !std::is_constructible<V, std::in_place_type_t<NoConstructible>>::value,
        "no matching constructor");
    static_assert(!std::is_constructible<V, std::in_place_index_t<1>>::value,
                  "no matching constructor");
  }



#if !defined(TEST_VARIANT_HAS_NO_REFERENCES)
  {
    using V = std::variant<int, int &&>;
    static_assert(!std::is_constructible<V, int>::value, "ambiguous");
  }
  {
    using V = std::variant<int, const int &>;
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
#ifndef TEST_VARIANT_ALLOWS_NARROWING_CONVERSIONS
  {
    constexpr std::variant<unsigned, long> v(42);
    static_assert(v.index() == 1, "");
    static_assert(std::get<1>(v) == 42, "");
  }
#endif
  {
    std::variant<std::string, bool const> v = "foo";
    assert(v.index() == 0);
    assert(std::get<0>(v) == "foo");
  }
  {
    std::variant<bool volatile, std::unique_ptr<int>> v = nullptr;
    assert(v.index() == 1);
    assert(std::get<1>(v) == nullptr);
  }
  {
    std::variant<bool volatile const, int> v = true;
    assert(v.index() == 0);
    assert(std::get<0>(v));
  }
  {
    std::variant<RValueConvertibleFrom<int>> v1 = 42;
    assert(v1.index() == 0);

    int x = 42;
    std::variant<RValueConvertibleFrom<int>, AnyConstructible> v2 = x;
    assert(v2.index() == 1);
  }
#if !defined(TEST_VARIANT_HAS_NO_REFERENCES)
  {
    using V = std::variant<const int &, int &&, long>;
    static_assert(std::is_convertible<int &, V>::value, "must be implicit");
    int x = 42;
    V v(x);
    assert(v.index() == 0);
    assert(&std::get<0>(v) == &x);
  }
  {
    using V = std::variant<const int &, int &&, long>;
    static_assert(std::is_convertible<int, V>::value, "must be implicit");
    int x = 42;
    V v(std::move(x));
    assert(v.index() == 1);
    assert(&std::get<1>(v) == &x);
  }
#endif
}

struct BoomOnAnything {
  template <class T>
  constexpr BoomOnAnything(T) { static_assert(!std::is_same<T, T>::value, ""); }
};

void test_no_narrowing_check_for_class_types() {
  using V = std::variant<int, BoomOnAnything>;
  V v(42);
  assert(v.index() == 0);
  assert(std::get<0>(v) == 42);
}

struct Bar {};
struct Baz {};
void test_construction_with_repeated_types() {
  using V = std::variant<int, Bar, Baz, int, Baz, int, int>;
  static_assert(!std::is_constructible<V, int>::value, "");
  static_assert(!std::is_constructible<V, Baz>::value, "");
  // OK, the selected type appears only once and so it shouldn't
  // be affected by the duplicate types.
  static_assert(std::is_constructible<V, Bar>::value, "");
}

int main(int, char**) {
  test_T_ctor_basic();
  test_T_ctor_noexcept();
  test_T_ctor_sfinae();
  test_no_narrowing_check_for_class_types();
  test_construction_with_repeated_types();
  return 0;
}
