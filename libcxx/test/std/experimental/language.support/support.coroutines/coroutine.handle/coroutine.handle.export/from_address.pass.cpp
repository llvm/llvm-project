// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

// <experimental/coroutine>

// template <class Promise = void>
// struct coroutine_handle;

// static coroutine_handle from_address(void*) noexcept

#include <experimental/coroutine>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

namespace coro = std::experimental;

template <class C>
void do_test() {
  {
    C c = C::from_address(nullptr);
    static_assert(noexcept(C::from_address(nullptr)), "");
    // FIXME: Should the return type not be 'C'?
    static_assert(std::is_same<decltype(C::from_address(nullptr)), C>::value, "");
    assert(c.address() == nullptr);
  }
  {
    char dummy = 42;
    C c = C::from_address((void*)&dummy);
    assert(c.address() == &dummy);
  }
}

int main(int, char**)
{
  do_test<coro::coroutine_handle<>>();
  do_test<coro::coroutine_handle<int>>();

  return 0;
}
