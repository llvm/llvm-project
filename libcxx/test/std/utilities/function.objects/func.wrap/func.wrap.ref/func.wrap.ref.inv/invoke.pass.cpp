//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// R operator()(ArgTypes... args) const noexcept(noex);

#include <cassert>
#include <concepts>
#include <functional>
#include <utility>
#include <type_traits>

#include "test_macros.h"
#include "MoveOnly.h"

template <class T, class... Args>
concept ConstInvocable = requires(const T t, Args... args) {
  { t(std::forward<Args>(args)...) };
};

template <class T, class... Args>
concept ConstNoexceptInvocable = requires(const T t, Args... args) {
  { t(std::forward<Args>(args)...) } noexcept;
};

static_assert(ConstInvocable<std::function_ref<void()>>);
static_assert(!ConstNoexceptInvocable<std::function_ref<void()>>);

static_assert(ConstInvocable<std::function_ref<void() noexcept>>);
static_assert(ConstNoexceptInvocable<std::function_ref<void() noexcept>>);

static_assert(ConstInvocable<std::function_ref<void() const>>);
static_assert(!ConstNoexceptInvocable<std::function_ref<void() const>>);

static_assert(ConstInvocable<std::function_ref<void() const noexcept>>);
static_assert(ConstNoexceptInvocable<std::function_ref<void() const noexcept>>);

struct S {
  int data = 42;
  int operator()(int& x) const noexcept { return data + x; }
  int operator()(const int& x) const noexcept { return data + x + 1; }
  int operator()(int&& x) const noexcept { return data + x + 2; }
  int operator()(const int&& x) const noexcept { return data + x + 3; }
};

struct S2 {
  double operator()(int x, int y, int z) noexcept { return x + y + z; }

  double operator()(int x, int y, int z) const noexcept { return x + y + z + 1; }
};

struct Big {
  char c[128];
};

struct S3Ref {
  char operator()(Big& b) const noexcept { return b.c[0]; }
};

struct S3Value {
  char operator()(Big b) const noexcept { return b.c[0]; }
};

void test_default() {
  {
    std::function_ref<void()> f = [] {};
    f();
    static_assert(std::is_void_v<decltype(f())>);
  }
  {
    // reference
    int x = 42;
    std::function_ref<int&(int&)> f(std::cw<[](int& i) -> int& { return i; }>);
    std::same_as<int&> decltype(auto) r = f(x);
    assert(&r == &x);
  }
  {
    // Move only
    std::function_ref<MoveOnly(MoveOnly)> f(std::cw<[](MoveOnly mo) { return MoveOnly{mo.get() + 5}; }>);
    std::same_as<MoveOnly> decltype(auto) r = f(MoveOnly{1});
    assert(r.get() == 6);
  }
  {
    // different return type
    std::function_ref<int(int)> f(std::cw<[](int x) -> double { return x + 0.1; }>);
    std::same_as<int> decltype(auto) r = f(5);
    assert(r == 5);
  }
  {
    // Args overload resolution
    S s;
    int x = 1;
    std::function_ref<int(int&)> f(s);
    assert(f(x) == 43);

    std::function_ref<int(const int&)> g(s);
    assert(g(x) == 44);

    std::function_ref<int(int&&)> h(s);
    assert(h(std::move(x)) == 45);

    std::function_ref<int(const int&&)> i(s);
    assert(i(std::move(x)) == 46);
  }
  {
    // const overload
    S2 s;
    std::function_ref<double(int, int, int)> f(s);
    std::same_as<double> decltype(auto) r = f(1, 2, 3);
    assert(r == 6.0);

    std::same_as<double> decltype(auto) r2 = std::as_const(f)(1, 2, 3);
    assert(r2 == 6.0);

    const S2 s2;
    std::function_ref<double(int, int, int)> f2(s2);
    std::same_as<double> decltype(auto) r3 = f2(1, 2, 3);
    assert(r3 == 7.0);

    std::same_as<double> decltype(auto) r4 = std::as_const(f2)(1, 2, 3);
    assert(r4 == 7.0);
  }
  {
    // Big object passed
    Big b{};
    b.c[0] = 'a';
    std::function_ref<char(Big&)> f1(std::cw<S3Ref{}>);
    assert(f1(b) == 'a');
    std::function_ref<char(Big)> f2(std::cw<S3Value{}>);
    assert(f2(b) == 'a');
  }
}

void test_const() {
  {
    std::function_ref<void() const> f = [] {};
    f();
    static_assert(std::is_void_v<decltype(f())>);
  }
  {
    // reference
    int x = 42;
    std::function_ref<int&(int&) const> f(std::cw<[](int& i) -> int& { return i; }>);
    std::same_as<int&> decltype(auto) r = f(x);
    assert(&r == &x);
  }
  {
    // Move only
    std::function_ref<MoveOnly(MoveOnly) const> f(std::cw<[](MoveOnly mo) { return MoveOnly{mo.get() + 5}; }>);
    std::same_as<MoveOnly> decltype(auto) r = f(MoveOnly{1});
    assert(r.get() == 6);
  }
  {
    // different return type
    std::function_ref<int(int) const> f(std::cw<[](int x) -> double { return x + 0.1; }>);
    std::same_as<int> decltype(auto) r = f(5);
    assert(r == 5);
  }
  {
    // Args overload resolution
    S s;
    int x = 1;
    std::function_ref<int(int&) const> f(s);
    assert(f(x) == 43);

    std::function_ref<int(const int&) const> g(s);
    assert(g(x) == 44);

    std::function_ref<int(int&&) const> h(s);
    assert(h(std::move(x)) == 45);

    std::function_ref<int(const int&&) const> i(s);
    assert(i(std::move(x)) == 46);
  }
  {
    // const overload
    S2 s;
    std::function_ref<double(int, int, int) const> f(s);
    std::same_as<double> decltype(auto) r = f(1, 2, 3);
    assert(r == 7.0);

    std::same_as<double> decltype(auto) r2 = std::as_const(f)(1, 2, 3);
    assert(r2 == 7.0);

    const S2 s2;
    std::function_ref<double(int, int, int) const> f2(s2);
    std::same_as<double> decltype(auto) r3 = f2(1, 2, 3);
    assert(r3 == 7.0);

    std::same_as<double> decltype(auto) r4 = std::as_const(f2)(1, 2, 3);
    assert(r4 == 7.0);
  }
  {
    // Big object passed
    Big b{};
    b.c[0] = 'a';
    std::function_ref<char(Big&) const> f1(std::cw<S3Ref{}>);
    assert(f1(b) == 'a');
    std::function_ref<char(Big) const> f2(std::cw<S3Value{}>);
    assert(f2(b) == 'a');
  }
}

void test_noexcept() {
  {
    std::function_ref<void() noexcept> f = [] noexcept {};
    f();
    static_assert(std::is_void_v<decltype(f())>);
  }
  {
    // reference
    int x = 42;
    std::function_ref<int&(int&) noexcept> f(std::cw<[](int& i) noexcept -> int& { return i; }>);
    std::same_as<int&> decltype(auto) r = f(x);
    assert(&r == &x);
  }
  {
    // Move only
    std::function_ref<MoveOnly(MoveOnly) noexcept> f(
        std::cw<[](MoveOnly mo) noexcept { return MoveOnly{mo.get() + 5}; }>);
    std::same_as<MoveOnly> decltype(auto) r = f(MoveOnly{1});
    assert(r.get() == 6);
  }
  {
    // different return type
    std::function_ref<int(int) noexcept> f(std::cw<[](int x) noexcept -> double { return x + 0.1; }>);
    std::same_as<int> decltype(auto) r = f(5);
    assert(r == 5);
  }
  {
    // Args overload resolution
    S s;
    int x = 1;
    std::function_ref<int(int&) noexcept> f(s);
    assert(f(x) == 43);

    std::function_ref<int(const int&) noexcept> g(s);
    assert(g(x) == 44);

    std::function_ref<int(int&&) noexcept> h(s);
    assert(h(std::move(x)) == 45);

    std::function_ref<int(const int&&) noexcept> i(s);
    assert(i(std::move(x)) == 46);
  }
  {
    // const overload
    S2 s;
    std::function_ref<double(int, int, int) noexcept> f(s);
    std::same_as<double> decltype(auto) r = f(1, 2, 3);
    assert(r == 6.0);

    std::same_as<double> decltype(auto) r2 = std::as_const(f)(1, 2, 3);
    assert(r2 == 6.0);

    const S2 s2;
    std::function_ref<double(int, int, int) noexcept> f2(s2);
    std::same_as<double> decltype(auto) r3 = f2(1, 2, 3);
    assert(r3 == 7.0);

    std::same_as<double> decltype(auto) r4 = std::as_const(f2)(1, 2, 3);
    assert(r4 == 7.0);
  }
  {
    // Big object passed
    Big b{};
    b.c[0] = 'a';
    std::function_ref<char(Big&) noexcept> f1(std::cw<S3Ref{}>);
    assert(f1(b) == 'a');
    std::function_ref<char(Big) noexcept> f2(std::cw<S3Value{}>);
    assert(f2(b) == 'a');
  }
}

void test_const_noexcept() {
  {
    std::function_ref<void() const noexcept> f = [] noexcept {};
    f();
    static_assert(std::is_void_v<decltype(f())>);
  }
  {
    // reference
    int x = 42;
    std::function_ref<int&(int&) const noexcept> f(std::cw<[](int& i) noexcept -> int& { return i; }>);
    std::same_as<int&> decltype(auto) r = f(x);
    assert(&r == &x);
  }
  {
    // Move only
    std::function_ref<MoveOnly(MoveOnly) const noexcept> f(
        std::cw<[](MoveOnly mo) noexcept { return MoveOnly{mo.get() + 5}; }>);
    std::same_as<MoveOnly> decltype(auto) r = f(MoveOnly{1});
    assert(r.get() == 6);
  }
  {
    // different return type
    std::function_ref<int(int) const noexcept> f(std::cw<[](int x) noexcept -> double { return x + 0.1; }>);
    std::same_as<int> decltype(auto) r = f(5);
    assert(r == 5);
  }
  {
    // Args overload resolution
    S s;
    int x = 1;
    std::function_ref<int(int&) const noexcept> f(s);
    assert(f(x) == 43);

    std::function_ref<int(const int&) const noexcept> g(s);
    assert(g(x) == 44);

    std::function_ref<int(int&&) const noexcept> h(s);
    assert(h(std::move(x)) == 45);

    std::function_ref<int(const int&&) const noexcept> i(s);
    assert(i(std::move(x)) == 46);
  }
  {
    // const overload
    S2 s;
    std::function_ref<double(int, int, int) const noexcept> f(s);
    std::same_as<double> decltype(auto) r = f(1, 2, 3);
    assert(r == 7.0);

    std::same_as<double> decltype(auto) r2 = std::as_const(f)(1, 2, 3);
    assert(r2 == 7.0);

    const S2 s2;
    std::function_ref<double(int, int, int) const noexcept> f2(s2);
    std::same_as<double> decltype(auto) r3 = f2(1, 2, 3);
    assert(r3 == 7.0);

    std::same_as<double> decltype(auto) r4 = std::as_const(f2)(1, 2, 3);
    assert(r4 == 7.0);
  }
  {
    // Big object passed
    Big b{};
    b.c[0] = 'a';
    std::function_ref<char(Big&) const noexcept> f1(std::cw<S3Ref{}>);
    assert(f1(b) == 'a');
    std::function_ref<char(Big) const noexcept> f2(std::cw<S3Value{}>);
    assert(f2(b) == 'a');
  }
}

void test() {
  test_default();
  test_const();
  test_noexcept();
  test_const_noexcept();
}

int main(int, char**) {
  test();
  return 0;
}
