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

// constexpr bool operator==(coroutine_handle<> x, coroutine_handle<> y) noexcept;

#include <coroutine>
#include <type_traits>
#include <utility>
#include <cstdint>
#include <cassert>

#include "test_macros.h"

template <class C>
void do_test(int *LHSVal, int *RHSVal) {
  const C LHS = C::from_address(LHSVal);
  const C RHS = C::from_address(RHSVal);
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
  int i;
  std::pair<int *, int *> const TestCases[] = {
      {nullptr, nullptr},
      {&i, &i},
      {nullptr, &i},
      {&i, nullptr}
  };
  for (auto& TC : TestCases) {
    do_test<std::coroutine_handle<>>(TC.first, TC.second);
    do_test<std::coroutine_handle<int>>(TC.first, TC.second);
  }

  return 0;
}
