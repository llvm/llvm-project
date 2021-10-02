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

// bool operator==(coroutine_handle<>, coroutine_handle<>) noexcept
// bool operator!=(coroutine_handle<>, coroutine_handle<>) noexcept

#include <experimental/coroutine>
#include <type_traits>
#include <utility>
#include <cstdint>
#include <cassert>

#include "test_macros.h"

namespace coro = std::experimental;

template <class C>
void do_test(uintptr_t LHSVal, uintptr_t RHSVal) {
  const C LHS = C::from_address(reinterpret_cast<void*>(LHSVal));
  const C RHS = C::from_address(reinterpret_cast<void*>(RHSVal));
  const bool ExpectIsEqual = (LHSVal == RHSVal);
  assert((LHS == RHS) == ExpectIsEqual);
  assert((RHS == LHS) == ExpectIsEqual);
  assert((LHS != RHS) == !ExpectIsEqual);
  assert((RHS != LHS) == !ExpectIsEqual);
  {
    static_assert(noexcept(LHS == RHS), "");
    static_assert(noexcept(LHS != RHS), "");
    ASSERT_SAME_TYPE(decltype(LHS == RHS), bool);
    ASSERT_SAME_TYPE(decltype(LHS != RHS), bool);
  }
}

int main(int, char**)
{
  std::pair<uintptr_t, uintptr_t> const TestCases[] = {
      {0, 0},
      {16, 16},
      {0, 16},
      {16, 0}
  };
  for (auto& TC : TestCases) {
    do_test<coro::coroutine_handle<>>(TC.first, TC.second);
    do_test<coro::coroutine_handle<int>>(TC.first, TC.second);
  }

  return 0;
}
