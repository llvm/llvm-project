//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <variant>
// template <class R, class Visitor, class... Variants>
// constexpr R visit(Visitor&& vis, Variants&&... vars);

#include <cassert>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

#include "test_macros.h"
#include "variant_test_helpers.h"

template <typename ReturnType>
void test_call_operator_forwarding() {
  using Fn = ForwardingCallObject;
  Fn obj{};
  const Fn &cobj = obj;
  { // test call operator forwarding - no variant
    std::visit<ReturnType>(obj);
    assert(Fn::check_call<>(CT_NonConst | CT_LValue));
    std::visit<ReturnType>(cobj);
    assert(Fn::check_call<>(CT_Const | CT_LValue));
    std::visit<ReturnType>(std::move(obj));
    assert(Fn::check_call<>(CT_NonConst | CT_RValue));
    std::visit<ReturnType>(std::move(cobj));
    assert(Fn::check_call<>(CT_Const | CT_RValue));
  }
  { // test call operator forwarding - single variant, single arg
    using V = std::variant<int>;
    V v(42);
    std::visit<ReturnType>(obj, v);
    assert(Fn::check_call<int &>(CT_NonConst | CT_LValue));
    std::visit<ReturnType>(cobj, v);
    assert(Fn::check_call<int &>(CT_Const | CT_LValue));
    std::visit<ReturnType>(std::move(obj), v);
    assert(Fn::check_call<int &>(CT_NonConst | CT_RValue));
    std::visit<ReturnType>(std::move(cobj), v);
    assert(Fn::check_call<int &>(CT_Const | CT_RValue));
  }
  { // test call operator forwarding - single variant, multi arg
    using V = std::variant<int, long, double>;
    V v(42l);
    std::visit<ReturnType>(obj, v);
    assert(Fn::check_call<long &>(CT_NonConst | CT_LValue));
    std::visit<ReturnType>(cobj, v);
    assert(Fn::check_call<long &>(CT_Const | CT_LValue));
    std::visit<ReturnType>(std::move(obj), v);
    assert(Fn::check_call<long &>(CT_NonConst | CT_RValue));
    std::visit<ReturnType>(std::move(cobj), v);
    assert(Fn::check_call<long &>(CT_Const | CT_RValue));
  }
  { // test call operator forwarding - multi variant, multi arg
    using V = std::variant<int, long, double>;
    using V2 = std::variant<int *, std::string>;
    V v(42l);
    V2 v2("hello");
    std::visit<int>(obj, v, v2);
    assert((Fn::check_call<long &, std::string &>(CT_NonConst | CT_LValue)));
    std::visit<ReturnType>(cobj, v, v2);
    assert((Fn::check_call<long &, std::string &>(CT_Const | CT_LValue)));
    std::visit<ReturnType>(std::move(obj), v, v2);
    assert((Fn::check_call<long &, std::string &>(CT_NonConst | CT_RValue)));
    std::visit<ReturnType>(std::move(cobj), v, v2);
    assert((Fn::check_call<long &, std::string &>(CT_Const | CT_RValue)));
  }
  {
    using V = std::variant<int, long, double, std::string>;
    V v1(42l), v2("hello"), v3(101), v4(1.1);
    std::visit<ReturnType>(obj, v1, v2, v3, v4);
    assert((Fn::check_call<long &, std::string &, int &, double &>(CT_NonConst | CT_LValue)));
    std::visit<ReturnType>(cobj, v1, v2, v3, v4);
    assert((Fn::check_call<long &, std::string &, int &, double &>(CT_Const | CT_LValue)));
    std::visit<ReturnType>(std::move(obj), v1, v2, v3, v4);
    assert((Fn::check_call<long &, std::string &, int &, double &>(CT_NonConst | CT_RValue)));
    std::visit<ReturnType>(std::move(cobj), v1, v2, v3, v4);
    assert((Fn::check_call<long &, std::string &, int &, double &>(CT_Const | CT_RValue)));
  }
  {
    using V = std::variant<int, long, double, int*, std::string>;
    V v1(42l), v2("hello"), v3(nullptr), v4(1.1);
    std::visit<ReturnType>(obj, v1, v2, v3, v4);
    assert((Fn::check_call<long &, std::string &, int *&, double &>(CT_NonConst | CT_LValue)));
    std::visit<ReturnType>(cobj, v1, v2, v3, v4);
    assert((Fn::check_call<long &, std::string &, int *&, double &>(CT_Const | CT_LValue)));
    std::visit<ReturnType>(std::move(obj), v1, v2, v3, v4);
    assert((Fn::check_call<long &, std::string &, int *&, double &>(CT_NonConst | CT_RValue)));
    std::visit<ReturnType>(std::move(cobj), v1, v2, v3, v4);
    assert((Fn::check_call<long &, std::string &, int *&, double &>(CT_Const | CT_RValue)));
  }
}

template <typename ReturnType>
void test_argument_forwarding() {
  using Fn = ForwardingCallObject;
  Fn obj{};
  const auto Val = CT_LValue | CT_NonConst;
  { // single argument - value type
    using V = std::variant<int>;
    V v(42);
    const V &cv = v;
    std::visit<ReturnType>(obj, v);
    assert(Fn::check_call<int &>(Val));
    std::visit<ReturnType>(obj, cv);
    assert(Fn::check_call<const int &>(Val));
    std::visit<ReturnType>(obj, std::move(v));
    assert(Fn::check_call<int &&>(Val));
    std::visit<ReturnType>(obj, std::move(cv));
    assert(Fn::check_call<const int &&>(Val));
  }
#if !defined(TEST_VARIANT_HAS_NO_REFERENCES)
  { // single argument - lvalue reference
    using V = std::variant<int &>;
    int x = 42;
    V v(x);
    const V &cv = v;
    std::visit<ReturnType>(obj, v);
    assert(Fn::check_call<int &>(Val));
    std::visit<ReturnType>(obj, cv);
    assert(Fn::check_call<int &>(Val));
    std::visit<ReturnType>(obj, std::move(v));
    assert(Fn::check_call<int &>(Val));
    std::visit<ReturnType>(obj, std::move(cv));
    assert(Fn::check_call<int &>(Val));
  }
  { // single argument - rvalue reference
    using V = std::variant<int &&>;
    int x = 42;
    V v(std::move(x));
    const V &cv = v;
    std::visit<ReturnType>(obj, v);
    assert(Fn::check_call<int &>(Val));
    std::visit<ReturnType>(obj, cv);
    assert(Fn::check_call<int &>(Val));
    std::visit<ReturnType>(obj, std::move(v));
    assert(Fn::check_call<int &&>(Val));
    std::visit<ReturnType>(obj, std::move(cv));
    assert(Fn::check_call<int &&>(Val));
  }
#endif
  { // multi argument - multi variant
    using V = std::variant<int, std::string, long>;
    V v1(42), v2("hello"), v3(43l);
    std::visit<ReturnType>(obj, v1, v2, v3);
    assert((Fn::check_call<int &, std::string &, long &>(Val)));
    std::visit<ReturnType>(obj, std::as_const(v1), std::as_const(v2), std::move(v3));
    assert((Fn::check_call<const int &, const std::string &, long &&>(Val)));
  }
  {
    using V = std::variant<int, long, double, std::string>;
    V v1(42l), v2("hello"), v3(101), v4(1.1);
    std::visit<ReturnType>(obj, v1, v2, v3, v4);
    assert((Fn::check_call<long &, std::string &, int &, double &>(Val)));
    std::visit<ReturnType>(obj, std::as_const(v1), std::as_const(v2), std::move(v3), std::move(v4));
    assert((Fn::check_call<const long &, const std::string &, int &&, double &&>(Val)));
  }
  {
    using V = std::variant<int, long, double, int*, std::string>;
    V v1(42l), v2("hello"), v3(nullptr), v4(1.1);
    std::visit<ReturnType>(obj, v1, v2, v3, v4);
    assert((Fn::check_call<long &, std::string &, int *&, double &>(Val)));
    std::visit<ReturnType>(obj, std::as_const(v1), std::as_const(v2), std::move(v3), std::move(v4));
    assert((Fn::check_call<const long &, const std::string &, int *&&, double &&>(Val)));
  }
}

template <typename ReturnType>
void test_return_type() {
  using Fn = ForwardingCallObject;
  Fn obj{};
  const Fn &cobj = obj;
  { // test call operator forwarding - no variant
    static_assert(std::is_same_v<decltype(std::visit<ReturnType>(obj)), ReturnType>);
    static_assert(std::is_same_v<decltype(std::visit<ReturnType>(cobj)), ReturnType>);
    static_assert(std::is_same_v<decltype(std::visit<ReturnType>(std::move(obj))), ReturnType>);
    static_assert(std::is_same_v<decltype(std::visit<ReturnType>(std::move(cobj))), ReturnType>);
  }
  { // test call operator forwarding - single variant, single arg
    using V = std::variant<int>;
    V v(42);
    static_assert(std::is_same_v<decltype(std::visit<ReturnType>(obj, v)), ReturnType>);
    static_assert(std::is_same_v<decltype(std::visit<ReturnType>(cobj, v)), ReturnType>);
    static_assert(std::is_same_v<decltype(std::visit<ReturnType>(std::move(obj), v)), ReturnType>);
    static_assert(std::is_same_v<decltype(std::visit<ReturnType>(std::move(cobj), v)), ReturnType>);
  }
  { // test call operator forwarding - single variant, multi arg
    using V = std::variant<int, long, double>;
    V v(42l);
    static_assert(std::is_same_v<decltype(std::visit<ReturnType>(obj, v)), ReturnType>);
    static_assert(std::is_same_v<decltype(std::visit<ReturnType>(cobj, v)), ReturnType>);
    static_assert(std::is_same_v<decltype(std::visit<ReturnType>(std::move(obj), v)), ReturnType>);
    static_assert(std::is_same_v<decltype(std::visit<ReturnType>(std::move(cobj), v)), ReturnType>);
  }
  { // test call operator forwarding - multi variant, multi arg
    using V = std::variant<int, long, double>;
    using V2 = std::variant<int *, std::string>;
    V v(42l);
    V2 v2("hello");
    static_assert(std::is_same_v<decltype(std::visit<ReturnType>(obj, v, v2)), ReturnType>);
    static_assert(std::is_same_v<decltype(std::visit<ReturnType>(cobj, v, v2)), ReturnType>);
    static_assert(std::is_same_v<decltype(std::visit<ReturnType>(std::move(obj), v, v2)), ReturnType>);
    static_assert(std::is_same_v<decltype(std::visit<ReturnType>(std::move(cobj), v, v2)), ReturnType>);
  }
  {
    using V = std::variant<int, long, double, std::string>;
    V v1(42l), v2("hello"), v3(101), v4(1.1);
    static_assert(std::is_same_v<decltype(std::visit<ReturnType>(obj, v1, v2, v3, v4)), ReturnType>);
    static_assert(std::is_same_v<decltype(std::visit<ReturnType>(cobj, v1, v2, v3, v4)), ReturnType>);
    static_assert(std::is_same_v<decltype(std::visit<ReturnType>(std::move(obj), v1, v2, v3, v4)), ReturnType>);
    static_assert(std::is_same_v<decltype(std::visit<ReturnType>(std::move(cobj), v1, v2, v3, v4)), ReturnType>);
  }
  {
    using V = std::variant<int, long, double, int*, std::string>;
    V v1(42l), v2("hello"), v3(nullptr), v4(1.1);
    static_assert(std::is_same_v<decltype(std::visit<ReturnType>(obj, v1, v2, v3, v4)), ReturnType>);
    static_assert(std::is_same_v<decltype(std::visit<ReturnType>(cobj, v1, v2, v3, v4)), ReturnType>);
    static_assert(std::is_same_v<decltype(std::visit<ReturnType>(std::move(obj), v1, v2, v3, v4)), ReturnType>);
    static_assert(std::is_same_v<decltype(std::visit<ReturnType>(std::move(cobj), v1, v2, v3, v4)), ReturnType>);
  }
}

void test_constexpr_void() {
  constexpr ReturnFirst obj{};
  constexpr ReturnArity aobj{};
  {
    using V = std::variant<int>;
    constexpr V v(42);
    static_assert((std::visit<void>(obj, v), 42) == 42, "");
  }
  {
    using V = std::variant<short, long, char>;
    constexpr V v(42l);
    static_assert((std::visit<void>(obj, v), 42) == 42, "");
  }
  {
    using V1 = std::variant<int>;
    using V2 = std::variant<int, char *, long long>;
    using V3 = std::variant<bool, int, int>;
    constexpr V1 v1;
    constexpr V2 v2(nullptr);
    constexpr V3 v3;
    static_assert((std::visit<void>(aobj, v1, v2, v3), 3) == 3, "");
  }
  {
    using V1 = std::variant<int>;
    using V2 = std::variant<int, char *, long long>;
    using V3 = std::variant<void *, int, int>;
    constexpr V1 v1;
    constexpr V2 v2(nullptr);
    constexpr V3 v3;
    static_assert((std::visit<void>(aobj, v1, v2, v3), 3) == 3, "");
  }
  {
    using V = std::variant<int, long, double, int *>;
    constexpr V v1(42l), v2(101), v3(nullptr), v4(1.1);
    static_assert((std::visit<void>(aobj, v1, v2, v3, v4), 4) == 4, "");
  }
  {
    using V = std::variant<int, long, double, long long, int *>;
    constexpr V v1(42l), v2(101), v3(nullptr), v4(1.1);
    static_assert((std::visit<void>(aobj, v1, v2, v3, v4), 4) == 4, "");
  }
}

void test_constexpr_int() {
  constexpr ReturnFirst obj{};
  constexpr ReturnArity aobj{};
  {
    using V = std::variant<int>;
    constexpr V v(42);
    static_assert(std::visit<int>(obj, v) == 42, "");
  }
  {
    using V = std::variant<short, long, char>;
    constexpr V v(42l);
    static_assert(std::visit<int>(obj, v) == 42, "");
  }
  {
    using V1 = std::variant<int>;
    using V2 = std::variant<int, char *, long long>;
    using V3 = std::variant<bool, int, int>;
    constexpr V1 v1;
    constexpr V2 v2(nullptr);
    constexpr V3 v3;
    static_assert(std::visit<int>(aobj, v1, v2, v3) == 3, "");
  }
  {
    using V1 = std::variant<int>;
    using V2 = std::variant<int, char *, long long>;
    using V3 = std::variant<void *, int, int>;
    constexpr V1 v1;
    constexpr V2 v2(nullptr);
    constexpr V3 v3;
    static_assert(std::visit<int>(aobj, v1, v2, v3) == 3, "");
  }
  {
    using V = std::variant<int, long, double, int *>;
    constexpr V v1(42l), v2(101), v3(nullptr), v4(1.1);
    static_assert(std::visit<int>(aobj, v1, v2, v3, v4) == 4, "");
  }
  {
    using V = std::variant<int, long, double, long long, int *>;
    constexpr V v1(42l), v2(101), v3(nullptr), v4(1.1);
    static_assert(std::visit<int>(aobj, v1, v2, v3, v4) == 4, "");
  }
}

template <typename ReturnType>
void test_exceptions() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  ReturnArity obj{};
  auto test = [&](auto &&... args) {
    try {
      std::visit<ReturnType>(obj, args...);
    } catch (const std::bad_variant_access &) {
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
  {
    using V = std::variant<int, MakeEmptyT>;
    using V2 = std::variant<long, std::string, void *>;
    V v;
    makeEmpty(v);
    V2 v2("hello");
    assert(test(v, v2));
  }
  {
    using V = std::variant<int, MakeEmptyT>;
    using V2 = std::variant<long, std::string, void *>;
    V v;
    makeEmpty(v);
    V2 v2("hello");
    assert(test(v2, v));
  }
  {
    using V = std::variant<int, MakeEmptyT>;
    using V2 = std::variant<long, std::string, void *, MakeEmptyT>;
    V v;
    makeEmpty(v);
    V2 v2;
    makeEmpty(v2);
    assert(test(v, v2));
  }
  {
    using V = std::variant<int, long, double, MakeEmptyT>;
    V v1(42l), v2(101), v3(202), v4(1.1);
    makeEmpty(v1);
    assert(test(v1, v2, v3, v4));
  }
  {
    using V = std::variant<int, long, double, long long, MakeEmptyT>;
    V v1(42l), v2(101), v3(202), v4(1.1);
    makeEmpty(v1);
    makeEmpty(v2);
    makeEmpty(v3);
    makeEmpty(v4);
    assert(test(v1, v2, v3, v4));
  }
#endif
}

// See https://bugs.llvm.org/show_bug.cgi?id=31916
template <typename ReturnType>
void test_caller_accepts_nonconst() {
  struct A {};
  struct Visitor {
    auto operator()(A&) {
      if constexpr (!std::is_void_v<ReturnType>)
      {
        return ReturnType{};
      }
    }
  };
  std::variant<A> v;
  std::visit<ReturnType>(Visitor{}, v);
}

void test_constexpr_explicit_side_effect() {
  auto test_lambda = [](int arg) constexpr {
    std::variant<int> v = 101;
    std::visit<void>([arg](int& x) constexpr { x = arg; }, v);
    return std::get<int>(v);
  };

  static_assert(test_lambda(202) == 202, "");
}

void test_derived_from_variant() {
  struct MyVariant : std::variant<short, long, float> {};

  std::visit<bool>(
      [](auto x) {
        assert(x == 42);
        return true;
      },
      MyVariant{42});
  std::visit<bool>(
      [](auto x) {
        assert(x == -1.3f);
        return true;
      },
      MyVariant{-1.3f});

  // Check that visit does not take index nor valueless_by_exception members from the base class.
  struct EvilVariantBase {
    int index;
    char valueless_by_exception;
  };

  struct EvilVariant1 : std::variant<int, long, double>,
                        std::tuple<int>,
                        EvilVariantBase {
    using std::variant<int, long, double>::variant;
  };

  std::visit<bool>(
      [](auto x) {
        assert(x == 12);
        return true;
      },
      EvilVariant1{12});
  std::visit<bool>(
      [](auto x) {
        assert(x == 12.3);
        return true;
      },
      EvilVariant1{12.3});

  // Check that visit unambiguously picks the variant, even if the other base has __impl member.
  struct ImplVariantBase {
    struct Callable {
      bool operator()() const { assert(false); return false; }
    };

    Callable __impl;
  };

  struct EvilVariant2 : std::variant<int, long, double>, ImplVariantBase {
    using std::variant<int, long, double>::variant;
  };

  std::visit<bool>(
      [](auto x) {
        assert(x == 12);
        return true;
      },
      EvilVariant2{12});
  std::visit<bool>(
      [](auto x) {
        assert(x == 12.3);
        return true;
      },
      EvilVariant2{12.3});
}

struct any_visitor {
  template <typename T>
  bool operator()(const T&) {
    return true;
  }
};

template <typename T, typename = decltype(std::visit<bool>(
                          std::declval<any_visitor&>(), std::declval<T>()))>
constexpr bool has_visit(int) {
  return true;
}

template <typename T>
constexpr bool has_visit(...) {
  return false;
}

void test_sfinae() {
  struct BadVariant : std::variant<short>, std::variant<long, float> {};

  static_assert(has_visit<std::variant<int> >(int()));
  static_assert(!has_visit<BadVariant>(int()));
}

int main(int, char**) {
  test_call_operator_forwarding<void>();
  test_argument_forwarding<void>();
  test_return_type<void>();
  test_constexpr_void();
  test_exceptions<void>();
  test_caller_accepts_nonconst<void>();
  test_call_operator_forwarding<int>();
  test_argument_forwarding<int>();
  test_return_type<int>();
  test_constexpr_int();
  test_exceptions<int>();
  test_caller_accepts_nonconst<int>();
  test_constexpr_explicit_side_effect();
  test_derived_from_variant();
  test_sfinae();

  return 0;
}
