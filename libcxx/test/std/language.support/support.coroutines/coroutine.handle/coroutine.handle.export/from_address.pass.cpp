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

// static coroutine_handle from_address(void*) noexcept

#include <coroutine>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

template <class C>
void do_test() {
  {
    C c = C::from_address(nullptr);
    static_assert(noexcept(C::from_address(nullptr)), "");
    static_assert(std::is_same<decltype(C::from_address(nullptr)), C>::value, "");
    assert(c.address() == nullptr);
  }
  {
    char dummy = 42;
    C c = C::from_address((void*)&dummy);
    assert(c.address() == &dummy);
  }
  {
    C::from_address((int*)nullptr);
    C::from_address((void*)nullptr);
    C::from_address((char*)nullptr);
  }
  {
    char dummy = 42;
    C c = C::from_address(&dummy);
    int* p = (int*)c.address();
    std::coroutine_handle<> c2 = std::coroutine_handle<>::from_address(p);
    assert(c2 == c);
  }
}

int main(int, char**)
{
  do_test<std::coroutine_handle<>>();
  do_test<std::coroutine_handle<int>>();

  return 0;
}
