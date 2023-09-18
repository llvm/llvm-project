//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <coroutine>

// template <class Promise = void>
// struct coroutine_handle;

// constexpr coroutine_handle() noexcept
// constexpr coroutine_handle(nullptr_t) noexcept

#include <coroutine>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

template <class C>
constexpr bool do_test() {
  static_assert(std::is_nothrow_constructible<C>::value, "");
  static_assert(std::is_nothrow_constructible<C, std::nullptr_t>::value, "");
  {
    C c;
    assert(c.address() == nullptr);
  }
  {
    C c = C(nullptr);
    assert(c.address() == nullptr);
  }
  return true;
}

int main(int, char**)
{
  do_test<std::coroutine_handle<>>();
  do_test<std::coroutine_handle<int>>();
  static_assert(do_test<std::coroutine_handle<>>());
  static_assert(do_test<std::coroutine_handle<int>>());

  return 0;
}
