//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// <experimental/coroutine>

// template <class Promise = void>
// struct coroutine_handle;

// coroutine_handle& operator=(nullptr_t) noexcept

#include <experimental/coroutine>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

namespace coro = std::experimental;

template <class C>
void do_test() {
  int dummy = 42;
  void* dummy_h = &dummy;
  {
    C c; ((void)c);
    static_assert(std::is_nothrow_assignable<C&, std::nullptr_t>::value, "");
    static_assert(!std::is_assignable<C&, void*>::value, "");
  }
  {
    C c = C::from_address(dummy_h);
    assert(c.address() == &dummy);
    c = nullptr;
    assert(c.address() == nullptr);
    c = nullptr;
    assert(c.address() == nullptr);
  }
  {
    C c;
    C& cr = (c = nullptr);
    assert(&c == &cr);
  }
}

int main(int, char**)
{
  do_test<coro::coroutine_handle<>>();
  do_test<coro::coroutine_handle<int>>();

  return 0;
}
