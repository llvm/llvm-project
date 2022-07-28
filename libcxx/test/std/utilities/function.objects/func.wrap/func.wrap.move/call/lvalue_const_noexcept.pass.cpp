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
#include <utility>

#include "test_macros.h"
#include "../common.h"

struct S {
  void func() const& noexcept {}
};

static_assert(std::is_invocable_v<std::move_only_function<void() const & noexcept>&>);
static_assert(std::is_invocable_v<std::move_only_function<void() const & noexcept>>);
static_assert(std::is_invocable_v<std::move_only_function<void() const & noexcept>&&>);
static_assert(std::is_invocable_v<std::move_only_function<void() const & noexcept> const&>);
static_assert(std::is_invocable_v<std::move_only_function<void() const & noexcept> const >);
static_assert(std::is_invocable_v<std::move_only_function<void() const & noexcept> const&&>);

void test() {
  {
    called                                            = false;
    std::move_only_function<void() const& noexcept> f = &call_func;
    f();
    assert(called);
  }
  {
    called                                            = false;
    std::move_only_function<void() const& noexcept> f = TriviallyDestructible{};
    f();
    assert(called);
  }
  {
    called                                            = false;
    std::move_only_function<void() const& noexcept> f = TriviallyDestructibleSqueezeFit{};
    f();
    assert(called);
  }
  {
    called                                            = false;
    std::move_only_function<void() const& noexcept> f = TriviallyDestructibleTooLarge{};
    f();
    assert(called);
  }
#ifdef TEST_COMPILER_CLANG
  {
    called                                            = false;
    std::move_only_function<void() const& noexcept> f = TriviallyRelocatable{};
    f();
    assert(called);
  }
  {
    called                                            = false;
    std::move_only_function<void() const& noexcept> f = TriviallyRelocatableSqueezeFit{};
    f();
    assert(called);
  }
  {
    called                                            = false;
    std::move_only_function<void() const& noexcept> f = TriviallyRelocatableTooLarge{};
    f();
    assert(called);
  }
#endif
  {
    called                                            = false;
    std::move_only_function<void() const& noexcept> f = NonTrivial{};
    f();
    assert(called);
  }
  {
    std::move_only_function<void(const S&) noexcept> f = &S::func;
    assert(f);
    LIBCPP_ASSERT(f.__get_status() == std::__move_only_function_storage::_Status::_TriviallyDestructible);
  }
  {
    decltype(&S::func) ptr                             = nullptr;
    std::move_only_function<void(const S&) noexcept> f = ptr;
    assert(!f);
    LIBCPP_ASSERT(f.__get_status() == std::__move_only_function_storage::_Status::_NotEngaged);
  }
  {
    CallType type;
    std::move_only_function<void() const& noexcept> f = CallTypeCheckerNoexcept{&type};
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

int main(int, char**) {
  test();

  return 0;
}
