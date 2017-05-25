// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11
// REQUIRES: fcoroutines-ts

// RUN: %build -fcoroutines-ts
// RUN: %run

// <experimental/coroutine>

// template <class Promise = void>
// struct coroutine_handle;

// void destroy() const

#include <experimental/coroutine>
#include <type_traits>
#include <memory>
#include <utility>
#include <cstdint>
#include <cassert>

#include "test_macros.h"

namespace coro = std::experimental;

template <class Promise>
void do_test(coro::coroutine_handle<Promise> const& H) {
  // FIXME Add a runtime test
  {
    ASSERT_SAME_TYPE(decltype(H.destroy()), void);
    ASSERT_NOT_NOEXCEPT(H.destroy());
  }
}

int main()
{
  do_test(coro::coroutine_handle<>{});
  do_test(coro::coroutine_handle<int>{});
}
