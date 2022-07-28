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
  void func() & noexcept {}
};

static_assert(std::is_invocable_v<std::move_only_function<void() & noexcept>&>);
static_assert(!std::is_invocable_v<std::move_only_function<void() & noexcept>>);
static_assert(!std::is_invocable_v<std::move_only_function<void() & noexcept>&&>);
static_assert(!std::is_invocable_v<std::move_only_function<void() & noexcept> const&>);
static_assert(!std::is_invocable_v<std::move_only_function<void() & noexcept> const >);
static_assert(!std::is_invocable_v<std::move_only_function<void() & noexcept> const&&>);

void test() {
  {
    called                                      = false;
    std::move_only_function<void()& noexcept> f = &call_func;
    f();
    assert(called);
  }
  {
    called                                      = false;
    std::move_only_function<void()& noexcept> f = TriviallyDestructible{};
    f();
    assert(called);
  }
  {
    called                                      = false;
    std::move_only_function<void()& noexcept> f = TriviallyDestructibleSqueezeFit{};
    f();
    assert(called);
  }
  {
    called                                      = false;
    std::move_only_function<void()& noexcept> f = TriviallyDestructibleTooLarge{};
    f();
    assert(called);
  }
#ifdef TEST_COMPILER_CLANG
  {
    called                                      = false;
    std::move_only_function<void()& noexcept> f = TriviallyRelocatable{};
    f();
    assert(called);
  }
  {
    called                                      = false;
    std::move_only_function<void()& noexcept> f = TriviallyRelocatableSqueezeFit{};
    f();
    assert(called);
  }
  {
    called                                      = false;
    std::move_only_function<void()& noexcept> f = TriviallyRelocatableTooLarge{};
    f();
    assert(called);
  }
#endif
  {
    called                                      = false;
    std::move_only_function<void()& noexcept> f = NonTrivial{};
    f();
    assert(called);
  }
  {
    std::move_only_function<void(S&) noexcept> f = &S::func;
    assert(f);
    LIBCPP_ASSERT(f.__get_status() == std::__move_only_function_storage::_Status::_TriviallyDestructible);
  }
  {
    decltype(&S::func) ptr                       = nullptr;
    std::move_only_function<void(S&) noexcept> f = ptr;
    assert(!f);
    LIBCPP_ASSERT(f.__get_status() == std::__move_only_function_storage::_Status::_NotEngaged);
  }
  {
    CallType type;
    std::move_only_function<void()& noexcept> f = CallTypeCheckerNoexcept{&type};
    f();
    assert(type == CallType::LValue);
  }
}

int main(int, char**) {
  test();

  return 0;
}
