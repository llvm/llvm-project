//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23
// The tested functionality needs deducing this.
// UNSUPPORTED: clang-16 || clang-17
// XFAIL: apple-clang

// <variant>

// class variant;

// template<class Self, class Visitor>
//   constexpr decltype(auto) visit(this Self&&, Visitor&&); // since C++26

#include <cassert>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include "test_macros.h"
#include "variant_test_helpers.h"

void test_call_operator_forwarding() {
  using Fn = ForwardingCallObject;
  Fn obj{};
  const Fn& cobj = obj;

  { // test call operator forwarding - single variant, single arg
    using V = std::variant<int>;
    V v(42);

    v.visit(obj);
    assert(Fn::check_call<int&>(CT_NonConst | CT_LValue));
    v.visit(cobj);
    assert(Fn::check_call<int&>(CT_Const | CT_LValue));
    v.visit(std::move(obj));
    assert(Fn::check_call<int&>(CT_NonConst | CT_RValue));
    v.visit(std::move(cobj));
    assert(Fn::check_call<int&>(CT_Const | CT_RValue));
  }
  { // test call operator forwarding - single variant, multi arg
    using V = std::variant<int, long, double>;
    V v(42L);

    v.visit(obj);
    assert(Fn::check_call<long&>(CT_NonConst | CT_LValue));
    v.visit(cobj);
    assert(Fn::check_call<long&>(CT_Const | CT_LValue));
    v.visit(std::move(obj));
    assert(Fn::check_call<long&>(CT_NonConst | CT_RValue));
    v.visit(std::move(cobj));
    assert(Fn::check_call<long&>(CT_Const | CT_RValue));
  }
}

// Applies to non-member `std::visit` only.
void test_argument_forwarding() {
  using Fn = ForwardingCallObject;
  Fn obj{};
  const auto val = CT_LValue | CT_NonConst;

  { // single argument - value type
    using V = std::variant<int>;
    V v(42);
    const V& cv = v;

    v.visit(obj);
    assert(Fn::check_call<int&>(val));
    cv.visit(obj);
    assert(Fn::check_call<const int&>(val));
    std::move(v).visit(obj);
    assert(Fn::check_call<int&&>(val));
    std::move(cv).visit(obj);
    assert(Fn::check_call<const int&&>(val));
  }
#if !defined(TEST_VARIANT_HAS_NO_REFERENCES)
  { // single argument - lvalue reference
    using V = std::variant<int&>;
    int x   = 42;
    V v(x);
    const V& cv = v;

    v.visit(obj);
    assert(Fn::check_call<int&>(val));
    cv.visit(obj);
    assert(Fn::check_call<int&>(val));
    std::move(v).visit(obj);
    assert(Fn::check_call<int&>(val));
    std::move(cv).visit(obj);
    assert(Fn::check_call<int&>(val));
    assert(false);
  }
  { // single argument - rvalue reference
    using V = std::variant<int&&>;
    int x   = 42;
    V v(std::move(x));
    const V& cv = v;

    v.visit(obj);
    assert(Fn::check_call<int&>(val));
    cvstd::visit(obj);
    assert(Fn::check_call<int&>(val));
    std::move(v).visit(obj);
    assert(Fn::check_call<int&&>(val));
    std::move(cv).visit(obj);
    assert(Fn::check_call<int&&>(val));
  }
#endif
}

void test_return_type() {
  using Fn = ForwardingCallObject;
  Fn obj{};
  const Fn& cobj = obj;

  { // test call operator forwarding - single variant, single arg
    using V = std::variant<int>;
    V v(42);

    static_assert(std::is_same_v<decltype(v.visit(obj)), Fn&>);
    static_assert(std::is_same_v<decltype(v.visit(cobj)), const Fn&>);
    static_assert(std::is_same_v<decltype(v.visit(std::move(obj))), Fn&&>);
    static_assert(std::is_same_v<decltype(v.visit(std::move(cobj))), const Fn&&>);
  }
  { // test call operator forwarding - single variant, multi arg
    using V = std::variant<int, long, double>;
    V v(42L);

    static_assert(std::is_same_v<decltype(v.visit(obj)), Fn&>);
    static_assert(std::is_same_v<decltype(v.visit(cobj)), const Fn&>);
    static_assert(std::is_same_v<decltype(v.visit(std::move(obj))), Fn&&>);
    static_assert(std::is_same_v<decltype(v.visit(std::move(cobj))), const Fn&&>);
  }
}

void test_constexpr() {
  constexpr ReturnFirst obj{};

  {
    using V = std::variant<int>;
    constexpr V v(42);

    static_assert(v.visit(obj) == 42);
  }
  {
    using V = std::variant<short, long, char>;
    constexpr V v(42L);

    static_assert(v.visit(obj) == 42);
  }
}

void test_exceptions() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  ReturnArity obj{};

  auto test = [&](auto&& v) {
    try {
      v.visit(obj);
    } catch (const std::bad_variant_access&) {
      return true;
    } catch (...) {
    }
    return false;
  };

  {
    using V = std::variant<int, MakeEmptyT>;
    V v;
    makeEmpty(v);

    assert(test(v));
  }
#endif
}

// See https://llvm.org/PR31916
void test_caller_accepts_nonconst() {
  struct A {};
  struct Visitor {
    void operator()(A&) {}
  };
  std::variant<A> v;

  v.visit(Visitor{});
}

struct MyVariant : std::variant<short, long, float> {};

namespace std {
template <std::size_t Index>
void get(const MyVariant&) {
  assert(false);
}
} // namespace std

void test_derived_from_variant() {
  auto v1        = MyVariant{42};
  const auto cv1 = MyVariant{142};

  v1.visit([](auto x) { assert(x == 42); });
  cv1.visit([](auto x) { assert(x == 142); });
  MyVariant{-1.25f}.visit([](auto x) { assert(x == -1.25f); });
  std::move(v1).visit([](auto x) { assert(x == 42); });
  std::move(cv1).visit([](auto x) { assert(x == 142); });

  // Check that visit does not take index nor valueless_by_exception members from the base class.
  struct EvilVariantBase {
    int index;
    char valueless_by_exception;
  };

  struct EvilVariant1 : std::variant<int, long, double>, std::tuple<int>, EvilVariantBase {
    using std::variant<int, long, double>::variant;
  };

  EvilVariant1{12}.visit([](auto x) { assert(x == 12); });
  EvilVariant1{12.3}.visit([](auto x) { assert(x == 12.3); });

  // Check that visit unambiguously picks the variant, even if the other base has __impl member.
  struct ImplVariantBase {
    struct Callable {
      bool operator()() const {
        assert(false);
        return false;
      }
    };

    Callable __impl;
  };

  struct EvilVariant2 : std::variant<int, long, double>, ImplVariantBase {
    using std::variant<int, long, double>::variant;
  };

  EvilVariant2{12}.visit([](auto x) { assert(x == 12); });
  EvilVariant2{12.3}.visit([](auto x) { assert(x == 12.3); });
}

int main(int, char**) {
  test_call_operator_forwarding();
  test_argument_forwarding();
  test_return_type();
  test_constexpr();
  test_exceptions();
  test_caller_accepts_nonconst();
  test_derived_from_variant();

  return 0;
}
