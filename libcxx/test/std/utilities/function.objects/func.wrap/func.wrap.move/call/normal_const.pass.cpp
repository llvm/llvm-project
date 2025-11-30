//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// GCC doesn't support [[clang::trivial_abi]] currently, which we want to use on
// move_only_function.
// UNSUPPORTED: gcc

#include <cassert>
#include <functional>
#include <utility>

#include "test_macros.h"
#include "../common.h"

struct S {
  void func() const noexcept {}
};

static_assert(std::is_invocable_v<std::move_only_function<void() const>&>);
static_assert(std::is_invocable_v<std::move_only_function<void() const>>);
static_assert(std::is_invocable_v<std::move_only_function<void() const>&&>);
static_assert(std::is_invocable_v<std::move_only_function<void() const> const&>);
static_assert(std::is_invocable_v<std::move_only_function<void() const> const >);
static_assert(std::is_invocable_v<std::move_only_function<void() const> const&&>);

void test() {
  {
    called                                  = false;
    std::move_only_function<void() const> f = &call_func;
    f();
    assert(called);
  }
  {
    called                                  = false;
    std::move_only_function<void() const> f = TriviallyDestructible{};
    f();
    assert(called);
  }
  {
    called                                  = false;
    std::move_only_function<void() const> f = TriviallyDestructibleTooLarge{};
    f();
    assert(called);
  }
  {
    called                                  = false;
    std::move_only_function<void() const> f = NonTrivial{};
    f();
    assert(called);
  }
  {
    std::move_only_function<void(S) const> f = &S::func;
    assert(f);
  }
  {
    decltype(&S::func) ptr                   = nullptr;
    std::move_only_function<void(S) const> f = ptr;
    assert(!f);
  }
  {
    CallType type;
    std::move_only_function<void() const> f = CallTypeChecker{&type};
    f();
    assert(type == CallType::ConstLValue);
    type = CallType::None;
    std::as_const(f)();
    assert(type == CallType::ConstLValue);
    type = CallType::None;
    std::move(f)();
    assert(type == CallType::ConstLValue);
    type = CallType::None;
    std::move(std::as_const(f))();
    assert(type == CallType::ConstLValue);
  }
}

void test_return() {
  {
    called                                    = false;
    std::move_only_function<int(int) const> f = &get_val;
    assert(f(3) == 3);
    assert(!called);
  }
  {
    called                                    = false;
    std::move_only_function<int(int) const> f = TriviallyDestructible{};
    assert(f(3) == 3);
    assert(!called);
  }
  {
    called                                    = false;
    std::move_only_function<int(int) const> f = TriviallyDestructibleTooLarge{};
    assert(f(3) == 3);
    assert(!called);
  }
  {
    called                                    = false;
    std::move_only_function<int(int) const> f = NonTrivial{};
    assert(f(3) == 3);
    assert(!called);
  }
}

int main(int, char**) {
  test();
  test_return();

  return 0;
}
