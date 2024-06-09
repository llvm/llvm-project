//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// template<class F> function_ref(F* f) noexcept;

#include <cassert>
#include <functional>
#include <utility>
#include <type_traits>

#include "test_macros.h"

// Constraints:
// - is_function_v<F> is true, and
// - is-invocable-using<F> is true.

struct A {
  int i;
  void f() {}
  void operator()(auto...) const {}
};

// non-const noexcept(false)
static_assert(std::is_constructible_v<std::function_ref<void()>, void (*)()>);
static_assert(std::is_constructible_v<std::function_ref<void()>, void (*)() noexcept>);
static_assert(!std::is_constructible_v<std::function_ref<void()>, void*>);
static_assert(!std::is_constructible_v<std::function_ref<void()>, void (*)(int)>);
static_assert(!std::is_constructible_v<std::function_ref<void()>, void (A::*)()>);
static_assert(!std::is_constructible_v<std::function_ref<void(A*)>, void (A::*)()>);

// the constructor is noexcept
static_assert(std::is_nothrow_constructible_v<std::function_ref<void()>, void (*)()>);
static_assert(std::is_nothrow_constructible_v<std::function_ref<void()>, void (*)() noexcept>);

// non-const noexcept(true)
static_assert(std::is_constructible_v<std::function_ref<A() noexcept>, A (*)() noexcept>);
static_assert(!std::is_constructible_v<std::function_ref<A() noexcept>, A (*)()>);
static_assert(!std::is_constructible_v<std::function_ref<A() noexcept>, A*>);
static_assert(!std::is_constructible_v<std::function_ref<void() noexcept>, void (*)(int) noexcept>);
static_assert(!std::is_constructible_v<std::function_ref<void() noexcept>, void (A::*)() noexcept>);
static_assert(!std::is_constructible_v<std::function_ref<void(A*) noexcept>, void (A::*)() noexcept>);

// the constructor is noexcept
static_assert(std::is_nothrow_constructible_v<std::function_ref<A() noexcept>, A (*)() noexcept>);

// const noexcept(false)
static_assert(std::is_constructible_v<std::function_ref<void(int, double) const>, void (*)(int, double)>);
static_assert(std::is_constructible_v<std::function_ref<void(int, double) const>, void (*)(int, double) noexcept>);
static_assert(!std::is_constructible_v<std::function_ref<void(int, double) const>, void*>);
static_assert(!std::is_constructible_v<std::function_ref<void(int, double) const>, void (*)(int, A)>);
static_assert(!std::is_constructible_v<std::function_ref<void(int, double) const>, void (A::*)()>);
static_assert(!std::is_constructible_v<std::function_ref<void(A*) const>, void (A::*)()>);

// the constructor is noexcept
static_assert(std::is_nothrow_constructible_v<std::function_ref<void(int, double) const>, void (*)(int, double)>);
static_assert(
    std::is_nothrow_constructible_v<std::function_ref<void(int, double) const>, void (*)(int, double) noexcept>);

// const noexcept(true)
static_assert(
    std::is_constructible_v<std::function_ref<void(int, double) const noexcept>, void (*)(int, double) noexcept>);
static_assert(!std::is_constructible_v<std::function_ref<void(int, double) const noexcept>, void (*)(int, double)>);
static_assert(!std::is_constructible_v<std::function_ref<void(int, double) const noexcept>, void*>);
static_assert(!std::is_constructible_v<std::function_ref<void(int, double) const noexcept>, void (*)(int, A) noexcept>);
static_assert(!std::is_constructible_v<std::function_ref<void(int, double) const noexcept>, void (A::*)() noexcept>);
static_assert(!std::is_constructible_v<std::function_ref<void(A*) const>, void (A::*)() noexcept>);

// the constructor is noexcept
static_assert(std::is_nothrow_constructible_v<std::function_ref<void(int, double) const noexcept>,
                                              void (*)(int, double) noexcept>);

int fn() { return 42; }

int fn_maythrow(int i, A a) { return i - a.i; }
int fn_noexcept(int i, A a) noexcept { return i + a.i; }

void foo(int) {}
void bar(int) noexcept {}
struct Int {
  int i;
  Int(int ii) noexcept : i(ii) {}
};

int needs_conversion(Int x, Int y, Int z) noexcept { return x.i + y.i + z.i; }

void test() {
  {
    // simple case
    std::function_ref<int()> f(&fn);
    assert(f() == 42);
  }
  {
    // explicit(false)
    std::function_ref<int()> f = &fn;
    assert(f() == 42);
  }
  {
    std::function_ref<int(int, A)> f(&fn_noexcept);
    assert(f(4, A{5}) == 9);
  }
  {
    // noexcept
    std::function_ref<int(int, A) noexcept> f(&fn_noexcept);
    assert(f(4, A{5}) == 9);
  }
  {
    // const
    auto l = [](int x, int y, int z) { return x + y - z; };
    std::function_ref<int(int, int, int) const> f(+l);
    assert(f(2, 3, 4) == 1);
  }
  {
    // const noexcept
    std::function_ref<int(int, A) const noexcept> f(&fn_noexcept);
    assert(f(4, A{5}) == 9);
  }

  {
    std::function_ref<Int(int, int, int)> f(&needs_conversion);
    assert(f(1, 2, 3).i == 6);

    std::function_ref<Int(int, int, int) const> f2(&needs_conversion);
    assert(f2(1, 2, 3).i == 6);

    std::function_ref<Int(int, int, int) noexcept> f3(&needs_conversion);
    assert(f3(1, 2, 3).i == 6);

    std::function_ref<Int(int, int, int) const noexcept> f4(&needs_conversion);
    assert(f4(1, 2, 3).i == 6);

    {
      std::function_ref r1 = foo;
      std::function_ref r2 = bar;
      r1                   = r2; // ok
    }
  }
}

int main(int, char**) {
  test();
  return 0;
}
