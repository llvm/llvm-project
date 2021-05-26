//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// template<class T, class U>
// concept invocable;

#include <chrono>
#include <concepts>
#include <memory>
#include <random>
#include <type_traits>

template <class R, class... Args>
[[nodiscard]] constexpr bool check_invocable() {
  constexpr bool result = std::invocable<R(Args...), Args...>;
  static_assert(std::invocable<R(Args...) noexcept, Args...> == result);
  static_assert(std::invocable<R (*)(Args...), Args...> == result);
  static_assert(std::invocable<R (*)(Args...) noexcept, Args...> == result);
  static_assert(std::invocable<R (&)(Args...), Args...> == result);
  static_assert(std::invocable<R (&)(Args...) noexcept, Args...> == result);

  return result;
}

static_assert(check_invocable<void>());
static_assert(check_invocable<void, int>());
static_assert(check_invocable<void, int&>());
static_assert(check_invocable<void, int*, double>());
static_assert(check_invocable<int>());
static_assert(check_invocable<int, int[]>());

struct S;
static_assert(check_invocable<int, int S::*, std::nullptr_t>());
static_assert(check_invocable<int, int (S::*)(), int (S::*)(int), int>());
static_assert(std::invocable<void (*)(int const&), int&>);
static_assert(std::invocable<void (*)(int const&), int&&>);
static_assert(std::invocable<void (*)(int volatile&), int&>);
static_assert(std::invocable<void (*)(int const volatile&), int&>);

static_assert(!std::invocable<void(), int>);
static_assert(!std::invocable<void(int)>);
static_assert(!std::invocable<void(int*), double*>);
static_assert(!std::invocable<void (*)(int&), double*>);
static_assert(std::invocable<int S::*, std::unique_ptr<S> >);
static_assert(std::invocable<int S::*, std::shared_ptr<S> >);
static_assert(!std::invocable<void (*)(int&&), int&>);
static_assert(!std::invocable<void (*)(int&&), int const&>);

static_assert(!std::invocable<void>);
static_assert(!std::invocable<void*>);
static_assert(!std::invocable<int>);
static_assert(!std::invocable<int&>);
static_assert(!std::invocable<int&&>);

namespace function_objects {
struct function_object {
  void operator()();
};
static_assert(std::invocable<function_object>);
static_assert(!std::invocable<function_object const>);
static_assert(!std::invocable<function_object volatile>);
static_assert(!std::invocable<function_object const volatile>);
static_assert(std::invocable<function_object&>);
static_assert(!std::invocable<function_object const&>);
static_assert(!std::invocable<function_object volatile&>);
static_assert(!std::invocable<function_object const volatile&>);

struct const_function_object {
  void operator()(int) const;
};
static_assert(std::invocable<const_function_object, int>);
static_assert(std::invocable<const_function_object const, int>);
static_assert(!std::invocable<const_function_object volatile, int>);
static_assert(!std::invocable<const_function_object const volatile, int>);
static_assert(std::invocable<const_function_object&, int>);
static_assert(std::invocable<const_function_object const&, int>);
static_assert(!std::invocable<const_function_object volatile&, int>);
static_assert(!std::invocable<const_function_object const volatile&, int>);

struct volatile_function_object {
  void operator()(int, int) volatile;
};
static_assert(std::invocable<volatile_function_object, int, int>);
static_assert(!std::invocable<volatile_function_object const, int, int>);
static_assert(std::invocable<volatile_function_object volatile, int, int>);
static_assert(
    !std::invocable<volatile_function_object const volatile, int, int>);
static_assert(std::invocable<volatile_function_object&, int, int>);
static_assert(!std::invocable<volatile_function_object const&, int, int>);
static_assert(std::invocable<volatile_function_object volatile&, int, int>);
static_assert(
    !std::invocable<volatile_function_object const volatile&, int, int>);

struct cv_function_object {
  void operator()(int[]) const volatile;
};
static_assert(std::invocable<cv_function_object, int*>);
static_assert(std::invocable<cv_function_object const, int*>);
static_assert(std::invocable<cv_function_object volatile, int*>);
static_assert(std::invocable<cv_function_object const volatile, int*>);
static_assert(std::invocable<cv_function_object&, int*>);
static_assert(std::invocable<cv_function_object const&, int*>);
static_assert(std::invocable<cv_function_object volatile&, int*>);
static_assert(std::invocable<cv_function_object const volatile&, int*>);

struct lvalue_function_object {
  void operator()() &;
};
static_assert(!std::invocable<lvalue_function_object>);
static_assert(!std::invocable<lvalue_function_object const>);
static_assert(!std::invocable<lvalue_function_object volatile>);
static_assert(!std::invocable<lvalue_function_object const volatile>);
static_assert(std::invocable<lvalue_function_object&>);
static_assert(!std::invocable<lvalue_function_object const&>);
static_assert(!std::invocable<lvalue_function_object volatile&>);
static_assert(!std::invocable<lvalue_function_object const volatile&>);

struct lvalue_const_function_object {
  void operator()(int) const&;
};
static_assert(std::invocable<lvalue_const_function_object, int>);
static_assert(std::invocable<lvalue_const_function_object const, int>);
static_assert(!std::invocable<lvalue_const_function_object volatile, int>);
static_assert(
    !std::invocable<lvalue_const_function_object const volatile, int>);
static_assert(std::invocable<lvalue_const_function_object&, int>);
static_assert(std::invocable<lvalue_const_function_object const&, int>);
static_assert(!std::invocable<lvalue_const_function_object volatile&, int>);
static_assert(
    !std::invocable<lvalue_const_function_object const volatile&, int>);

struct lvalue_volatile_function_object {
  void operator()(int, int) volatile&;
};
static_assert(!std::invocable<lvalue_volatile_function_object, int, int>);
static_assert(!std::invocable<lvalue_volatile_function_object const, int, int>);
static_assert(
    !std::invocable<lvalue_volatile_function_object volatile, int, int>);
static_assert(
    !std::invocable<lvalue_volatile_function_object const volatile, int, int>);
static_assert(std::invocable<lvalue_volatile_function_object&, int, int>);
static_assert(
    !std::invocable<lvalue_volatile_function_object const&, int, int>);
static_assert(
    std::invocable<lvalue_volatile_function_object volatile&, int, int>);
static_assert(
    !std::invocable<lvalue_volatile_function_object const volatile&, int, int>);

struct lvalue_cv_function_object {
  void operator()(int[]) const volatile&;
};
static_assert(!std::invocable<lvalue_cv_function_object, int*>);
static_assert(!std::invocable<lvalue_cv_function_object const, int*>);
static_assert(!std::invocable<lvalue_cv_function_object volatile, int*>);
static_assert(!std::invocable<lvalue_cv_function_object const volatile, int*>);
static_assert(std::invocable<lvalue_cv_function_object&, int*>);
static_assert(std::invocable<lvalue_cv_function_object const&, int*>);
static_assert(std::invocable<lvalue_cv_function_object volatile&, int*>);
static_assert(std::invocable<lvalue_cv_function_object const volatile&, int*>);
//
struct rvalue_function_object {
  void operator()() &&;
};
static_assert(std::invocable<rvalue_function_object>);
static_assert(!std::invocable<rvalue_function_object const>);
static_assert(!std::invocable<rvalue_function_object volatile>);
static_assert(!std::invocable<rvalue_function_object const volatile>);
static_assert(!std::invocable<rvalue_function_object&>);
static_assert(!std::invocable<rvalue_function_object const&>);
static_assert(!std::invocable<rvalue_function_object volatile&>);
static_assert(!std::invocable<rvalue_function_object const volatile&>);

struct rvalue_const_function_object {
  void operator()(int) const&&;
};
static_assert(std::invocable<rvalue_const_function_object, int>);
static_assert(std::invocable<rvalue_const_function_object const, int>);
static_assert(!std::invocable<rvalue_const_function_object volatile, int>);
static_assert(
    !std::invocable<rvalue_const_function_object const volatile, int>);
static_assert(!std::invocable<rvalue_const_function_object&, int>);
static_assert(!std::invocable<rvalue_const_function_object const&, int>);
static_assert(!std::invocable<rvalue_const_function_object volatile&, int>);
static_assert(
    !std::invocable<rvalue_const_function_object const volatile&, int>);

struct rvalue_volatile_function_object {
  void operator()(int, int) volatile&&;
};
static_assert(std::invocable<rvalue_volatile_function_object, int, int>);
static_assert(!std::invocable<rvalue_volatile_function_object const, int, int>);
static_assert(
    std::invocable<rvalue_volatile_function_object volatile, int, int>);
static_assert(
    !std::invocable<rvalue_volatile_function_object const volatile, int, int>);
static_assert(!std::invocable<rvalue_volatile_function_object&, int, int>);
static_assert(
    !std::invocable<rvalue_volatile_function_object const&, int, int>);
static_assert(
    !std::invocable<rvalue_volatile_function_object volatile&, int, int>);
static_assert(
    !std::invocable<rvalue_volatile_function_object const volatile&, int, int>);

struct rvalue_cv_function_object {
  void operator()(int[]) const volatile&&;
};
static_assert(std::invocable<rvalue_cv_function_object, int*>);
static_assert(std::invocable<rvalue_cv_function_object const, int*>);
static_assert(std::invocable<rvalue_cv_function_object volatile, int*>);
static_assert(std::invocable<rvalue_cv_function_object const volatile, int*>);
static_assert(!std::invocable<rvalue_cv_function_object&, int*>);
static_assert(!std::invocable<rvalue_cv_function_object const&, int*>);
static_assert(!std::invocable<rvalue_cv_function_object volatile&, int*>);
static_assert(!std::invocable<rvalue_cv_function_object const volatile&, int*>);

struct multiple_overloads {
  struct A {};
  struct B { B(int); };
  struct AB : A, B {};
  struct O {};
  void operator()(A) const;
  void operator()(B) const;
};
static_assert(std::invocable<multiple_overloads, multiple_overloads::A>);
static_assert(std::invocable<multiple_overloads, multiple_overloads::B>);
static_assert(std::invocable<multiple_overloads, int>);
static_assert(!std::invocable<multiple_overloads, multiple_overloads::AB>);
static_assert(!std::invocable<multiple_overloads, multiple_overloads::O>);
} // namespace function_objects

namespace pointer_to_member_functions {
// clang-format off
  template<class Member, class T, class... Args>
  [[nodiscard]] constexpr bool check_member_is_invocable()
  {
    constexpr bool result = std::invocable<Member, T, Args...>;
    using uncv_t = std::remove_cvref_t<T>;
    static_assert(std::invocable<Member, uncv_t*, Args...> == result);
    static_assert(std::invocable<Member, std::unique_ptr<uncv_t>, Args...> == result);
    static_assert(std::invocable<Member, std::reference_wrapper<uncv_t>, Args...> == result);
    static_assert(!std::invocable<Member, std::nullptr_t, Args...>);
    static_assert(!std::invocable<Member, int, Args...>);
    static_assert(!std::invocable<Member, int*, Args...>);
    static_assert(!std::invocable<Member, double*, Args...>);
    struct S2 {};
    static_assert(!std::invocable<Member, S2*, Args...>);
    return result;
  }
// clang-format on

static_assert(check_member_is_invocable<int S::*, S>());
static_assert(std::invocable<int S::*, S&>);
static_assert(std::invocable<int S::*, S const&>);
static_assert(std::invocable<int S::*, S volatile&>);
static_assert(std::invocable<int S::*, S const volatile&>);
static_assert(std::invocable<int S::*, S&&>);
static_assert(std::invocable<int S::*, S const&&>);
static_assert(std::invocable<int S::*, S volatile&&>);
static_assert(std::invocable<int S::*, S const volatile&&>);

static_assert(check_member_is_invocable<int (S::*)(int), S, int>());
static_assert(!check_member_is_invocable<int (S::*)(int), S>());
using unqualified = void (S::*)();
static_assert(std::invocable<unqualified, S&>);
static_assert(!std::invocable<unqualified, S const&>);
static_assert(!std::invocable<unqualified, S volatile&>);
static_assert(!std::invocable<unqualified, S const volatile&>);
static_assert(std::invocable<unqualified, S&&>);
static_assert(!std::invocable<unqualified, S const&&>);
static_assert(!std::invocable<unqualified, S volatile&&>);
static_assert(!std::invocable<unqualified, S const volatile&&>);

static_assert(check_member_is_invocable<int (S::*)(double) const, S, double>());
using const_qualified = void (S::*)() const;
static_assert(std::invocable<const_qualified, S&>);
static_assert(std::invocable<const_qualified, S const&>);
static_assert(!std::invocable<const_qualified, S volatile&>);
static_assert(!std::invocable<const_qualified, S const volatile&>);
static_assert(std::invocable<const_qualified, S&&>);
static_assert(std::invocable<const_qualified, S const&&>);
static_assert(!std::invocable<const_qualified, S volatile&&>);
static_assert(!std::invocable<const_qualified, S const volatile&&>);

static_assert(
    check_member_is_invocable<int (S::*)(double[]) volatile, S, double*>());
using volatile_qualified = void (S::*)() volatile;
static_assert(std::invocable<volatile_qualified, S&>);
static_assert(!std::invocable<volatile_qualified, S const&>);
static_assert(std::invocable<volatile_qualified, S volatile&>);
static_assert(!std::invocable<volatile_qualified, S const volatile&>);
static_assert(std::invocable<volatile_qualified, S&&>);
static_assert(!std::invocable<volatile_qualified, S const&&>);
static_assert(std::invocable<volatile_qualified, S volatile&&>);
static_assert(!std::invocable<volatile_qualified, S const volatile&&>);

static_assert(check_member_is_invocable<int (S::*)(int, S&) const volatile, S,
                                        int, S&>());
using cv_qualified = void (S::*)() const volatile;
static_assert(std::invocable<cv_qualified, S&>);
static_assert(std::invocable<cv_qualified, S const&>);
static_assert(std::invocable<cv_qualified, S volatile&>);
static_assert(std::invocable<cv_qualified, S const volatile&>);
static_assert(std::invocable<cv_qualified, S&&>);
static_assert(std::invocable<cv_qualified, S const&&>);
static_assert(std::invocable<cv_qualified, S volatile&&>);
static_assert(std::invocable<cv_qualified, S const volatile&&>);

static_assert(check_member_is_invocable<int (S::*)() &, S&>());
using lvalue_qualified = void (S::*)() &;
static_assert(std::invocable<lvalue_qualified, S&>);
static_assert(!std::invocable<lvalue_qualified, S const&>);
static_assert(!std::invocable<lvalue_qualified, S volatile&>);
static_assert(!std::invocable<lvalue_qualified, S const volatile&>);
static_assert(!std::invocable<lvalue_qualified, S&&>);
static_assert(!std::invocable<lvalue_qualified, S const&&>);
static_assert(!std::invocable<lvalue_qualified, S volatile&&>);
static_assert(!std::invocable<lvalue_qualified, S const volatile&&>);

static_assert(check_member_is_invocable<int (S::*)() const&, S>());
using lvalue_const_qualified = void (S::*)() const&;
static_assert(std::invocable<lvalue_const_qualified, S&>);
static_assert(std::invocable<lvalue_const_qualified, S const&>);
static_assert(!std::invocable<lvalue_const_qualified, S volatile&>);
static_assert(!std::invocable<lvalue_const_qualified, S const volatile&>);
static_assert(std::invocable<lvalue_const_qualified, S&&>);
static_assert(std::invocable<lvalue_const_qualified, S const&&>);
static_assert(!std::invocable<lvalue_const_qualified, S volatile&&>);
static_assert(!std::invocable<lvalue_const_qualified, S const volatile&&>);

static_assert(check_member_is_invocable<int (S::*)() volatile&, S&>());
using lvalue_volatile_qualified = void (S::*)() volatile&;
static_assert(std::invocable<lvalue_volatile_qualified, S&>);
static_assert(!std::invocable<lvalue_volatile_qualified, S const&>);
static_assert(std::invocable<lvalue_volatile_qualified, S volatile&>);
static_assert(!std::invocable<lvalue_volatile_qualified, S const volatile&>);
static_assert(!std::invocable<lvalue_volatile_qualified, S&&>);
static_assert(!std::invocable<lvalue_volatile_qualified, S const&&>);
static_assert(!std::invocable<lvalue_volatile_qualified, S volatile&&>);
static_assert(!std::invocable<lvalue_volatile_qualified, S const volatile&&>);

static_assert(check_member_is_invocable<int (S::*)() const volatile&, S&>());
using lvalue_cv_qualified = void (S::*)() const volatile&;
static_assert(std::invocable<lvalue_cv_qualified, S&>);
static_assert(std::invocable<lvalue_cv_qualified, S const&>);
static_assert(std::invocable<lvalue_cv_qualified, S volatile&>);
static_assert(std::invocable<lvalue_cv_qualified, S const volatile&>);
static_assert(!std::invocable<lvalue_cv_qualified, S&&>);
static_assert(!std::invocable<lvalue_cv_qualified, S const&&>);
static_assert(!std::invocable<lvalue_cv_qualified, S volatile&&>);
static_assert(!std::invocable<lvalue_cv_qualified, S const volatile&&>);

using rvalue_unqualified = void (S::*)() &&;
static_assert(!std::invocable<rvalue_unqualified, S&>);
static_assert(!std::invocable<rvalue_unqualified, S const&>);
static_assert(!std::invocable<rvalue_unqualified, S volatile&>);
static_assert(!std::invocable<rvalue_unqualified, S const volatile&>);
static_assert(std::invocable<rvalue_unqualified, S&&>);
static_assert(!std::invocable<rvalue_unqualified, S const&&>);
static_assert(!std::invocable<rvalue_unqualified, S volatile&&>);
static_assert(!std::invocable<rvalue_unqualified, S const volatile&&>);

using rvalue_const_unqualified = void (S::*)() const&&;
static_assert(!std::invocable<rvalue_const_unqualified, S&>);
static_assert(!std::invocable<rvalue_const_unqualified, S const&>);
static_assert(!std::invocable<rvalue_const_unqualified, S volatile&>);
static_assert(!std::invocable<rvalue_const_unqualified, S const volatile&>);
static_assert(std::invocable<rvalue_const_unqualified, S&&>);
static_assert(std::invocable<rvalue_const_unqualified, S const&&>);
static_assert(!std::invocable<rvalue_const_unqualified, S volatile&&>);
static_assert(!std::invocable<rvalue_const_unqualified, S const volatile&&>);

using rvalue_volatile_unqualified = void (S::*)() volatile&&;
static_assert(!std::invocable<rvalue_volatile_unqualified, S&>);
static_assert(!std::invocable<rvalue_volatile_unqualified, S const&>);
static_assert(!std::invocable<rvalue_volatile_unqualified, S volatile&>);
static_assert(!std::invocable<rvalue_volatile_unqualified, S const volatile&>);
static_assert(std::invocable<rvalue_volatile_unqualified, S&&>);
static_assert(!std::invocable<rvalue_volatile_unqualified, S const&&>);
static_assert(std::invocable<rvalue_volatile_unqualified, S volatile&&>);
static_assert(!std::invocable<rvalue_volatile_unqualified, S const volatile&&>);

using rvalue_cv_unqualified = void (S::*)() const volatile&&;
static_assert(!std::invocable<rvalue_cv_unqualified, S&>);
static_assert(!std::invocable<rvalue_cv_unqualified, S const&>);
static_assert(!std::invocable<rvalue_cv_unqualified, S volatile&>);
static_assert(!std::invocable<rvalue_cv_unqualified, S const volatile&>);
static_assert(std::invocable<rvalue_cv_unqualified, S&&>);
static_assert(std::invocable<rvalue_cv_unqualified, S const&&>);
static_assert(std::invocable<rvalue_cv_unqualified, S volatile&&>);
static_assert(std::invocable<rvalue_cv_unqualified, S const volatile&&>);
} // namespace pointer_to_member_functions

// std::invocable-specific
static_assert(std::invocable<std::uniform_int_distribution<>, std::mt19937_64&>);

// Check the concept with closure types
template<class F, class... Args>
constexpr bool is_invocable(F, Args&&...) {
  return std::invocable<F, Args...>;
}

static_assert(is_invocable([] {}));
static_assert(is_invocable([](int) {}, 0));
static_assert(is_invocable([](int) {}, 0L));
static_assert(!is_invocable([](int) {}, nullptr));
int i = 0;
static_assert(is_invocable([](int&) {}, i));
