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
  void func() && noexcept {}
};

static_assert(!std::is_invocable_v<std::move_only_function<void() &&>&>);
static_assert(std::is_invocable_v<std::move_only_function<void() &&>>);
static_assert(std::is_invocable_v<std::move_only_function<void() &&>&&>);
static_assert(!std::is_invocable_v<std::move_only_function<void() &&> const&>);
static_assert(!std::is_invocable_v<std::move_only_function<void() &&> const >);
static_assert(!std::is_invocable_v<std::move_only_function<void() &&> const&&>);

void test() {
  {
    called                              = false;
    std::move_only_function<void()&&> f = &call_func;
    std::move(f)();
    assert(called);
  }
  {
    called                              = false;
    std::move_only_function<void()&&> f = TriviallyDestructible{};
    std::move(f)();
    assert(called);
  }
  {
    called                              = false;
    std::move_only_function<void()&&> f = TriviallyDestructibleTooLarge{};
    std::move(f)();
    assert(called);
  }
  {
    called                              = false;
    std::move_only_function<void()&&> f = NonTrivial{};
    std::move(f)();
    assert(called);
  }
  {
    std::move_only_function<void(S)&&> f = &S::func;
    assert(f);
  }
  {
    decltype(&S::func) ptr               = nullptr;
    std::move_only_function<void(S)&&> f = ptr;
    assert(!f);
  }
  {
    CallType type;
    std::move_only_function<void()&&> f = CallTypeChecker{&type};
    type                                = CallType::None;
    std::move(f)();
    assert(type == CallType::RValue);
  }
}

void test_return() {
  {
    called                                = false;
    std::move_only_function<int(int)&&> f = &get_val;
    assert(std::move(f)(3) == 3);
    assert(!called);
  }
  {
    called                                = false;
    std::move_only_function<int(int)&&> f = TriviallyDestructible{};
    assert(std::move(f)(3) == 3);
    assert(!called);
  }
  {
    called                                = false;
    std::move_only_function<int(int)&&> f = TriviallyDestructibleTooLarge{};
    assert(std::move(f)(3) == 3);
    assert(!called);
  }
  {
    called                                = false;
    std::move_only_function<int(int)&&> f = NonTrivial{};
    assert(std::move(f)(3) == 3);
    assert(!called);
  }
}

int main(int, char**) {
  test();
  test_return();

  return 0;
}
