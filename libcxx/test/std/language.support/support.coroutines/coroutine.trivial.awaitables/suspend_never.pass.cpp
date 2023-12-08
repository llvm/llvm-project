//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <coroutine>
#include <type_traits>
#include <cassert>
#include <utility>

#include "test_macros.h"

constexpr bool check_suspend_constexpr() {
  std::suspend_never s{};
  const std::suspend_never scopy(s); ((void)scopy);
  std::suspend_never smove(std::move(s)); ((void)smove);
  s = scopy;
  s = std::move(smove);
  return true;
}

template<class T>
constexpr bool test_trivial_awaitable_constexpr(bool expected) {
  T t;
  assert(t.await_ready() == expected);
  t.await_suspend(nullptr);
  t.await_resume();
  return true;
}

int main(int, char**)
{
  using H = std::coroutine_handle<>;
  using S = std::suspend_never;
  H h{};
  S s{};
  S const& cs = s;
  {
    LIBCPP_STATIC_ASSERT(noexcept(s.await_ready()), "");
    static_assert(std::is_same<decltype(s.await_ready()), bool>::value, "");
    assert(s.await_ready() == true);
    assert(cs.await_ready() == true);
  }
  {
    LIBCPP_STATIC_ASSERT(noexcept(s.await_suspend(h)), "");
    static_assert(std::is_same<decltype(s.await_suspend(h)), void>::value, "");
    s.await_suspend(h);
    cs.await_suspend(h);
  }
  {
    LIBCPP_STATIC_ASSERT(noexcept(s.await_resume()), "");
    static_assert(std::is_same<decltype(s.await_resume()), void>::value, "");
    s.await_resume();
    cs.await_resume();
  }
  {
    static_assert(std::is_nothrow_default_constructible<std::suspend_never>::value, "");
    static_assert(std::is_nothrow_copy_constructible<std::suspend_never>::value, "");
    static_assert(std::is_nothrow_move_constructible<std::suspend_never>::value, "");
    static_assert(std::is_nothrow_copy_assignable<std::suspend_never>::value, "");
    static_assert(std::is_nothrow_move_assignable<std::suspend_never>::value, "");
    static_assert(std::is_trivially_copyable<std::suspend_never>::value, "");
    static_assert(check_suspend_constexpr(), "");
  }
  {
    static_assert(test_trivial_awaitable_constexpr<std::suspend_never>(true));
  }

  return 0;
}
