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

// namespace std {
//  template <class P> struct hash<coroutine_handle<P>>;
// }

#include <coroutine>
#include <type_traits>
#include <memory>
#include <utility>
#include <cstdint>
#include <cassert>
#include <functional>

#include "test_macros.h"

template <class C>
void do_test(int *LHSVal, int *RHSVal) {
  const size_t ExpectLHS = std::hash<void*>{}((LHSVal));
  const size_t ExpectRHS = std::hash<void*>{}((RHSVal));
  const C LHS = C::from_address((LHSVal));
  const C RHS = C::from_address((RHSVal));
  const std::hash<C> h;

  LIBCPP_ASSERT(h(LHS) == ExpectLHS);
  LIBCPP_ASSERT(h(RHS) == ExpectRHS);
  assert((h(LHS) == h(RHS)) == (LHSVal == RHSVal));
  {
    ASSERT_SAME_TYPE(decltype(h(LHS)), size_t);
    ASSERT_NOEXCEPT(std::hash<C>{}(LHS));
  }
}

int main(int, char**)
{
  int i, j;
  std::pair<int *, int *> const TestCases[] = {
      {nullptr, nullptr},
      {nullptr, &i},
      {&i, &i},
      {&i, &j}
  };
  for (auto& TC : TestCases) {
    do_test<std::coroutine_handle<>>(TC.first, TC.second);
    do_test<std::coroutine_handle<int>>(TC.first, TC.second);
  }

  return 0;
}
