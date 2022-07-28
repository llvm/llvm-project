//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <cassert>
#include <functional>

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
    std::move_only_function<void()&&> f = TriviallyDestructibleSqueezeFit{};
    std::move(f)();
    assert(called);
  }
  {
    called                              = false;
    std::move_only_function<void()&&> f = TriviallyDestructibleTooLarge{};
    std::move(f)();
    assert(called);
  }
#ifdef TEST_COMPILER_CLANG
  {
    called                              = false;
    std::move_only_function<void()&&> f = TriviallyRelocatable{};
    std::move(f)();
    assert(called);
  }
  {
    called                              = false;
    std::move_only_function<void()&&> f = TriviallyRelocatableSqueezeFit{};
    std::move(f)();
    assert(called);
  }
  {
    called                              = false;
    std::move_only_function<void()&&> f = TriviallyRelocatableTooLarge{};
    std::move(f)();
    assert(called);
  }
#endif
  {
    called                              = false;
    std::move_only_function<void()&&> f = NonTrivial{};
    std::move(f)();
    assert(called);
  }
  {
    std::move_only_function<void(S)&&> f = &S::func;
    assert(f);
    LIBCPP_ASSERT(f.__get_status() == std::__move_only_function_storage::_Status::_TriviallyDestructible);
  }
  {
    decltype(&S::func) ptr               = nullptr;
    std::move_only_function<void(S)&&> f = ptr;
    assert(!f);
    LIBCPP_ASSERT(f.__get_status() == std::__move_only_function_storage::_Status::_NotEngaged);
  }
  {
    CallType type;
    std::move_only_function<void()&&> f = CallTypeChecker{&type};
    type                                = CallType::None;
    std::move(f)();
    assert(type == CallType::RValue);
  }
}

int main(int, char**) {
  test();

  return 0;
}
