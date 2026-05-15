//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// template<class T> function_ref& operator=(T) = delete;

#include <cassert>
#include <functional>
#include <utility>
#include <type_traits>

#include "test_macros.h"

// Constraints:
// - T is not the same type as function_ref,
// - is_pointer_v<T> is false, and
// - T is not a specialization of constant_wrapper.

// non const noexcept(false)
static_assert(std::is_assignable_v<std::function_ref<void()>, std::function_ref<void()>>);
static_assert(std::is_assignable_v<std::function_ref<void()>, std::function_ref<void() const>>);
static_assert(std::is_assignable_v<std::function_ref<void()>, std::function_ref<void() noexcept>>);
static_assert(std::is_assignable_v<std::function_ref<void()>, std::function_ref<void() const noexcept>>);

static_assert(std::is_assignable_v<std::function_ref<void()>, void (*)()>);
static_assert(!std::is_assignable_v<std::function_ref<void()>, void (*)(int)>);

static_assert(std::is_assignable_v<std::function_ref<void()>, std::constant_wrapper<[] {}>>);
static_assert(!std::is_assignable_v<std::function_ref<void()>, std::constant_wrapper<[](int) {}>>);

// const noexcept(false)
static_assert(!std::is_assignable_v<std::function_ref<void() const>, std::function_ref<void()>>);
static_assert(std::is_assignable_v<std::function_ref<void() const>, std::function_ref<void() const>>);
static_assert(!std::is_assignable_v<std::function_ref<void() const>, std::function_ref<void() noexcept>>);
static_assert(std::is_assignable_v<std::function_ref<void() const>, std::function_ref<void() const noexcept>>);

static_assert(std::is_assignable_v<std::function_ref<void() const>, void (*)()>);
static_assert(!std::is_assignable_v<std::function_ref<void() const>, void (*)(int)>);

static_assert(std::is_assignable_v<std::function_ref<void() const>, std::constant_wrapper<[] { return 42; }>>);
static_assert(!std::is_assignable_v<std::function_ref<void() const>, std::constant_wrapper<[](int) { return 42; }>>);

// non-const noexcept(true)
static_assert(!std::is_assignable_v<std::function_ref<void() noexcept>, std::function_ref<void()>>);
static_assert(!std::is_assignable_v<std::function_ref<void() noexcept>, std::function_ref<void() const>>);
static_assert(std::is_assignable_v<std::function_ref<void() noexcept>, std::function_ref<void() noexcept>>);
static_assert(std::is_assignable_v<std::function_ref<void() noexcept>, std::function_ref<void() const noexcept>>);

static_assert(std::is_assignable_v<std::function_ref<void() noexcept>, void (*)() noexcept>);
static_assert(!std::is_assignable_v<std::function_ref<void() noexcept>, void (*)(int) noexcept>);

static_assert(std::is_assignable_v<std::function_ref<void() noexcept>, std::constant_wrapper<[] noexcept {} >>);
static_assert(!std::is_assignable_v<std::function_ref<void() noexcept>, std::constant_wrapper<[](int) noexcept {}>>);

// const noexcept(true)
static_assert(!std::is_assignable_v<std::function_ref<void() const noexcept>, std::function_ref<void()>>);
static_assert(!std::is_assignable_v<std::function_ref<void() const noexcept>, std::function_ref<void() const>>);
static_assert(!std::is_assignable_v<std::function_ref<void() const noexcept>, std::function_ref<void() noexcept>>);
static_assert(std::is_assignable_v<std::function_ref<void() const noexcept>, std::function_ref<void() const noexcept>>);

static_assert(std::is_assignable_v<std::function_ref<void() const noexcept>, void (*)() noexcept>);
static_assert(!std::is_assignable_v<std::function_ref<void() const noexcept>, void (*)(int) noexcept>);

static_assert(std::is_assignable_v<std::function_ref<void() const noexcept>, std::constant_wrapper<[] noexcept {}>>);
static_assert(
    !std::is_assignable_v<std::function_ref<void() const noexcept>, std::constant_wrapper<[](int) noexcept {}>>);

int forty_two() { return 42; }

void test() {
  {
    std::function_ref<int()> f(std::cw<[] { return 41; }>);
    f = std::function_ref<int()>(std::cw<[] { return 42; }>);
    assert(f() == 42);
  }
  {
    std::function_ref<int() > f(std::cw<[] { return 41; }>);
    f = &forty_two;
    assert(f() == 42);
  }
  {
    std::function_ref<int() > f(std::cw<[] { return 41; }>);
    f = std::cw<[] { return 42; }>;
    assert(f() == 42);
  }
  {
    // const
    std::function_ref<int() const> f(std::cw<[] { return 41; }>);
    f = std::cw<[] { return 42; }>;
    assert(f() == 42);
  }
  {
    // noexcept
    std::function_ref<int() noexcept> f(std::cw<[] noexcept { return 41; }>);
    f = std::cw<[] noexcept { return 42; }>;
    assert(f() == 42);
  }
  {
    // const noexcept
    std::function_ref<int() const noexcept> f(std::cw<[] noexcept { return 41; }>);
    f = std::cw<[] noexcept { return 42; }>;
    assert(f() == 42);
  }
}

int main(int, char**) {
  test();
  return 0;
}
