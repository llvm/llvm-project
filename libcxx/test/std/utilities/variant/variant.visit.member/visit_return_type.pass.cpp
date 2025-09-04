//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23
// The tested functionality needs deducing this.
// XFAIL: apple-clang

// <variant>

// class variant;

// template<class R, class Self, class Visitor>
//   constexpr R visit(this Self&&, Visitor&&);              // since C++26

#include <cassert>
#include <memory>
#include <string>
#include <type_traits>
#include <tuple>
#include <utility>
#include <variant>

#include "test_macros.h"
#include "variant_test_helpers.h"

template <class... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};

template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

void test_overload_ambiguity() {
  using V = std::variant<float, long, std::string>;
  using namespace std::string_literals;
  V v{"baba"s};

  v.visit(
      overloaded{[]([[maybe_unused]] auto x) { assert(false); }, [](const std::string& x) { assert(x == "baba"s); }});
  assert(std::get<std::string>(v) == "baba"s);

  // Test the constraint.
  v = std::move(v).visit<V>(overloaded{
      []([[maybe_unused]] auto x) {
        assert(false);
        return 0;
      },
      [](const std::string& x) {
        assert(x == "baba"s);
        return x + " zmt"s;
      }});
  assert(std::get<std::string>(v) == "baba zmt"s);
}

template <typename ReturnType>
void test_call_operator_forwarding() {
  using Fn = ForwardingCallObject;
  Fn obj{};
  const Fn& cobj = obj;

  { // test call operator forwarding - single variant, single arg
    using V = std::variant<int>;
    V v(42);

    v.visit<ReturnType>(obj);
    assert(Fn::check_call<int&>(CT_NonConst | CT_LValue));
    v.visit<ReturnType>(cobj);
    assert(Fn::check_call<int&>(CT_Const | CT_LValue));
    v.visit<ReturnType>(std::move(obj));
    assert(Fn::check_call<int&>(CT_NonConst | CT_RValue));
    v.visit<ReturnType>(std::move(cobj));
    assert(Fn::check_call<int&>(CT_Const | CT_RValue));
  }
  { // test call operator forwarding - single variant, multi arg
    using V = std::variant<int, long, double>;
    V v(42L);

    v.visit<ReturnType>(obj);
    assert(Fn::check_call<long&>(CT_NonConst | CT_LValue));
    v.visit<ReturnType>(cobj);
    assert(Fn::check_call<long&>(CT_Const | CT_LValue));
    v.visit<ReturnType>(std::move(obj));
    assert(Fn::check_call<long&>(CT_NonConst | CT_RValue));
    v.visit<ReturnType>(std::move(cobj));
    assert(Fn::check_call<long&>(CT_Const | CT_RValue));
  }
}

template <typename ReturnType>
void test_argument_forwarding() {
  using Fn = ForwardingCallObject;
  Fn obj{};
  const auto val = CT_LValue | CT_NonConst;

  { // single argument - value type
    using V = std::variant<int>;
    V v(42);
    const V& cv = v;

    v.visit<ReturnType>(obj);
    assert(Fn::check_call<int&>(val));
    cv.visit<ReturnType>(obj);
    assert(Fn::check_call<const int&>(val));
    std::move(v).visit<ReturnType>(obj);
    assert(Fn::check_call<int&&>(val));
    std::move(cv).visit<ReturnType>(obj);
    assert(Fn::check_call<const int&&>(val));
  }
}

template <typename ReturnType>
void test_return_type() {
  using Fn = ForwardingCallObject;
  Fn obj{};
  const Fn& cobj = obj;

  { // test call operator forwarding - no variant
    // non-member
    {
      static_assert(std::is_same_v<decltype(std::visit<ReturnType>(obj)), ReturnType>);
      static_assert(std::is_same_v<decltype(std::visit<ReturnType>(cobj)), ReturnType>);
      static_assert(std::is_same_v<decltype(std::visit<ReturnType>(std::move(obj))), ReturnType>);
      static_assert(std::is_same_v<decltype(std::visit<ReturnType>(std::move(cobj))), ReturnType>);
    }
  }
  { // test call operator forwarding - single variant, single arg
    using V = std::variant<int>;
    V v(42);

    static_assert(std::is_same_v<decltype(v.visit<ReturnType>(obj)), ReturnType>);
    static_assert(std::is_same_v<decltype(v.visit<ReturnType>(cobj)), ReturnType>);
    static_assert(std::is_same_v<decltype(v.visit<ReturnType>(std::move(obj))), ReturnType>);
    static_assert(std::is_same_v<decltype(v.visit<ReturnType>(std::move(cobj))), ReturnType>);
  }
  { // test call operator forwarding - single variant, multi arg
    using V = std::variant<int, long, double>;
    V v(42L);

    static_assert(std::is_same_v<decltype(v.visit<ReturnType>(obj)), ReturnType>);
    static_assert(std::is_same_v<decltype(v.visit<ReturnType>(cobj)), ReturnType>);
    static_assert(std::is_same_v<decltype(v.visit<ReturnType>(std::move(obj))), ReturnType>);
    static_assert(std::is_same_v<decltype(v.visit<ReturnType>(std::move(cobj))), ReturnType>);
  }
}

void test_constexpr_void() {
  constexpr ReturnFirst obj{};

  {
    using V = std::variant<int>;
    constexpr V v(42);

    static_assert((v.visit<void>(obj), 42) == 42);
  }
  {
    using V = std::variant<short, long, char>;
    constexpr V v(42L);

    static_assert((v.visit<void>(obj), 42) == 42);
  }
}

void test_constexpr_int() {
  constexpr ReturnFirst obj{};

  {
    using V = std::variant<int>;
    constexpr V v(42);

    static_assert(v.visit<int>(obj) == 42);
  }
  {
    using V = std::variant<short, long, char>;
    constexpr V v(42L);

    static_assert(v.visit<int>(obj) == 42);
  }
}

template <typename ReturnType>
void test_exceptions() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  ReturnArity obj{};

  auto test = [&](auto&& v) {
    try {
      v.template visit<ReturnType>(obj);
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
template <typename ReturnType>
void test_caller_accepts_nonconst() {
  struct A {};
  struct Visitor {
    auto operator()(A&) {
      if constexpr (!std::is_void_v<ReturnType>) {
        return ReturnType{};
      }
    }
  };
  std::variant<A> v;

  v.template visit<ReturnType>(Visitor{});
}

void test_constexpr_explicit_side_effect() {
  auto test_lambda = [](int arg) constexpr {
    std::variant<int> v = 101;

    {
      v.template visit<void>([arg](int& x) constexpr { x = arg; });
    }

    return std::get<int>(v);
  };

  static_assert(test_lambda(202) == 202);
}

void test_derived_from_variant() {
  struct MyVariant : std::variant<short, long, float> {};

  MyVariant{42}.template visit<bool>([](auto x) {
    assert(x == 42);
    return true;
  });
  MyVariant{-1.3f}.template visit<bool>([](auto x) {
    assert(x == -1.3f);
    return true;
  });

  // Check that visit does not take index nor valueless_by_exception members from the base class.
  struct EvilVariantBase {
    int index;
    char valueless_by_exception;
  };

  struct EvilVariant1 : std::variant<int, long, double>, std::tuple<int>, EvilVariantBase {
    using std::variant<int, long, double>::variant;
  };

  EvilVariant1{12}.template visit<bool>([](auto x) {
    assert(x == 12);
    return true;
  });
  EvilVariant1{12.3}.template visit<bool>([](auto x) {
    assert(x == 12.3);
    return true;
  });

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

  EvilVariant2{12}.template visit<bool>([](auto x) {
    assert(x == 12);
    return true;
  });
  EvilVariant2{12.3}.template visit<bool>([](auto x) {
    assert(x == 12.3);
    return true;
  });
}

int main(int, char**) {
  test_overload_ambiguity();
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

  return 0;
}
