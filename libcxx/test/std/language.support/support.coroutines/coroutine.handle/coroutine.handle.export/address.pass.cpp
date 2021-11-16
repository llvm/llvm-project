//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-coroutines

// <coroutine>

// template <class Promise = void>
// struct coroutine_handle;

// constexpr void* address() const noexcept

#include <coroutine>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

template <class C>
constexpr bool do_test() {
  {
    constexpr C c; ((void)c);
    static_assert(c.address() == nullptr, "");
  }
  {
    const C c = {}; ((void)c);
    ASSERT_NOEXCEPT(c.address());
    ASSERT_SAME_TYPE(decltype(c.address()), void*);
    assert(c.address() == nullptr);
  }
  {
    char dummy = 42;
    C c = C::from_address((void*)&dummy);
    assert(c.address() == &dummy);
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
