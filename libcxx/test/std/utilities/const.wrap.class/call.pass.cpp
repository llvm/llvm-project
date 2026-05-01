//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// constant_wrapper

// template<class... Args>
// static constexpr decltype(auto) operator()(Args&&... args) noexcept(see below);

#include <cassert>
#include <concepts>
#include <functional>
#include <type_traits>
#include <utility>

#include "helpers.h"
#include "MoveOnly.h"

struct MoveOnlyFn {
  constexpr MoveOnly operator()(const MoveOnly& m1, MoveOnly m2, MoveOnly&& m3) const {
    return MoveOnly(m1.get() + m2.get() + m3.get());
  }
};

constexpr bool fun_ptr(int i) { return i > 0; }

struct OverloadSet {
  constexpr int operator()(int) const { return 1; }

  constexpr int operator()(std::constant_wrapper<42>) const { return 2; }
};

struct ReturnNonStructural {
  constexpr NonStructural operator()(int i) const { return NonStructural{i}; }
};

struct CWOnly {
  constexpr int operator()(std::constant_wrapper<42>) const { return 42; }
};

constexpr int nothrow_call(int) noexcept { return 42; }

constexpr int throwing_call(int) { return 42; }

struct S {
  int member = 42;

  constexpr int mem_fun(int i) const { return member + i; }
};

constexpr S s;

// Let call-expr be constant_wrapper<INVOKE (value, remove_cvref_t<Args>::value...)>{} if all types
// in remove_cvref_t<Args>... satisfy constexpr-param and constant_wrapper<INVOKE (value, remove_-
// cvref_t<Args>::value...)> is a valid type, otherwise let call-expr be INVOKE (value, std::forward<Args>(args)...).
//
// Constraints: call-expr is a valid expression.
// Remarks: The exception specification is equivalent to noexcept(call-expr).

// clang-format off
static_assert(std::is_invocable_v<std::constant_wrapper<[] { return 42; }>>);
static_assert(!std::is_invocable_v<std::constant_wrapper<[] { return 42; }>, int>);
static_assert(!std::is_invocable_v<std::constant_wrapper<5>>);

static_assert(!std::is_invocable_v<std::constant_wrapper<std::plus<>{}>, int>);
static_assert(std::is_nothrow_invocable_v<std::constant_wrapper<std::plus<>{}>, int, int>);
static_assert(std::is_nothrow_invocable_v<std::constant_wrapper<std::plus<>{}>, std::constant_wrapper<42>, int>);
static_assert(std::is_nothrow_invocable_v<std::constant_wrapper<std::plus<>{}>, std::constant_wrapper<42>, std::constant_wrapper<42>>);

static_assert(std::is_nothrow_invocable_v<std::constant_wrapper<nothrow_call>, int>);
static_assert(std::is_nothrow_invocable_v<std::constant_wrapper<nothrow_call>, std::constant_wrapper<42>>);

static_assert(std::is_invocable_v<std::constant_wrapper<throwing_call>, int>);
static_assert(!std::is_nothrow_invocable_v<std::constant_wrapper<throwing_call>, int>);
static_assert(std::is_nothrow_invocable_v<std::constant_wrapper<throwing_call>, std::constant_wrapper<42>>,
              "the call expression is still nothrow because the constexpr path is taken");
// clang-format on

template <class T>
struct MustBeInt {
  static_assert(std::same_as<T, int>);
};

struct Poison {
  template <class T>
  constexpr auto operator()(T) const noexcept -> MustBeInt<T> {
    return {};
  }
};

constexpr bool test() {
  {
    // with runtime param
    using T                                 = std::constant_wrapper<std::plus<>{}>;
    std::same_as<int> decltype(auto) result = T::operator()(1, 2);
    assert(result == 3);
  }

  {
    // with runtime param and constexpr param
    using T                                 = std::constant_wrapper<std::plus<>{}>;
    std::same_as<int> decltype(auto) result = T::operator()(std::cw<1>, 2);
    assert(result == 3);
  }

  {
    // with only constexpr param
    using T                                                      = std::constant_wrapper<std::plus<>{}>;
    std::same_as<std::constant_wrapper<3>> decltype(auto) result = T::operator()(std::cw<1>, std::cw<2>);
    static_assert(result == 3);
  }

  {
    // nullary
    using T                                                       = std::constant_wrapper<[] { return 42; }>;
    std::same_as<std::constant_wrapper<42>> decltype(auto) result = T::operator()();
    static_assert(result == 42);
  }

  {
    // return void with runtime param
    using T = std::constant_wrapper<[](int) {}>;
    T::operator()(5);
    static_assert(std::same_as<void, decltype(T::operator()(5))>);
  }

  {
    // return void with constexpr param
    using T = std::constant_wrapper<[](int) {}>;
    T::operator()(std::cw<5>);
    static_assert(std::same_as<void, decltype(T::operator()(std::cw<5>))>);
  }

  {
    // nullary return void
    using T = std::constant_wrapper<[] {}>;
    T::operator()();
    static_assert(std::same_as<void, decltype(T::operator()())>);
  }

  {
    // move only
    using T = std::constant_wrapper<MoveOnlyFn{}>;
    MoveOnly m1(1), m2(2), m3(3);
    std::same_as<MoveOnly> decltype(auto) result = T::operator()(m1, std::move(m2), std::move(m3));
    assert(result.get() == 6);
  }

  {
    // function pointer
    using T                                  = std::constant_wrapper<fun_ptr>;
    std::same_as<bool> decltype(auto) result = T::operator()(5);
    assert(result);
  }

  {
    // function pointer with constexpr param
    using T                                                         = std::constant_wrapper<fun_ptr>;
    std::same_as<std::constant_wrapper<true>> decltype(auto) result = T::operator()(std::cw<5>);
    static_assert(result);
  }
  {
    // member ptr with runtime param
    using T = std::constant_wrapper<&S::member>;
    S s1;
    std::same_as<int&> decltype(auto) result = T::operator()(s1);
    assert(result == 42);
    assert(&result == &s1.member);
  }
  {
    // member ptr with constexpr param
    using T                                                       = std::constant_wrapper<&S::member>;
    std::same_as<std::constant_wrapper<42>> decltype(auto) result = T::operator()(std::cw<&s>);
    static_assert(result == 42);
  }
  {
    // member function ptr with runtime param
    using T = std::constant_wrapper<&S::mem_fun>;
    S s1;
    std::same_as<int> decltype(auto) result = T::operator()(s1, 8);
    assert(result == 50);
  }
  {
    // member function ptr with constexpr param
    using T                                                       = std::constant_wrapper<&S::mem_fun>;
    std::same_as<std::constant_wrapper<50>> decltype(auto) result = T::operator()(std::cw<&s>, std::cw<8>);
    static_assert(result == 50);
  }
  {
    // overload set
    // will always unwrap the constexpr params and call the non-constexpr overload
    using T                                  = std::constant_wrapper<OverloadSet{}>;
    std::same_as<int> decltype(auto) result1 = T::operator()(42);
    assert(result1 == 1);
    std::same_as<std::constant_wrapper<1>> decltype(auto) result2 = T::operator()(std::cw<42>);
    static_assert(result2 == 1);
  }

  {
    // return non-structural type
    using T                                           = std::constant_wrapper<ReturnNonStructural{}>;
    std::same_as<NonStructural> decltype(auto) result = T::operator()(5);
    assert(result.get() == 5);
  }

  {
    // return non-structural type with constexpr param
    using T                                           = std::constant_wrapper<ReturnNonStructural{}>;
    std::same_as<NonStructural> decltype(auto) result = T::operator()(std::cw<5>);
    assert(result.get() == 5);
  }

  {
    // cw only
    // the upwrapping case doesn't work so it falls back to the normal invoke path
    using T                                 = std::constant_wrapper<CWOnly{}>;
    std::same_as<int> decltype(auto) result = T::operator()(std::cw<42>);
    assert(result == 42);
  }

  {
    // just use the call operator
    assert(std::cw<[](int i) { return i + 1; }>(42) == 43);
    assert(std::cw<[](int i) { return i + 1; }>(std::cw<42>) == 43);
  }

  {
    // with integral_constant, will still call the constexpr path
    using T = std::constant_wrapper<std::plus<>{}>;
    std::integral_constant<int, 1> ic1;
    std::integral_constant<int, 2> ic2;
    std::same_as<std::constant_wrapper<3>> decltype(auto) result = T::operator()(ic1, ic2);
    static_assert(result == 3);
  }

  {
    using T = std::constant_wrapper<Poison{}>;
    [[maybe_unused]] std::same_as<std::constant_wrapper<MustBeInt<int>{}>> decltype(auto) result =
        T::operator()(std::cw<5>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
