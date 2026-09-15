//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// UNSUPPORTED: availability-stacktrace-missing

// (19.6.4.3) Observers [stacktrace.basic.obs]
//
//   const_reverse_iterator rbegin() const noexcept;
//   const_reverse_iterator rend() const noexcept;

#include <cassert>
#include <iterator>
#include <stacktrace>
#include "test_macros.h"

// Call chain is: main -> c -> b -> a -> stacktrace::current
TEST_NO_TAIL_CALLS TEST_NOINLINE std::stacktrace a() { return std::stacktrace::current(); }
TEST_NO_TAIL_CALLS TEST_NOINLINE std::stacktrace b() { return a(); }
TEST_NO_TAIL_CALLS TEST_NOINLINE std::stacktrace c() { return b(); }

int main(int, char**) {
  {
    std::stacktrace const st;
    static_assert(noexcept(st.rbegin()));
    static_assert(noexcept(st.rend()));
    static_assert(std::random_access_iterator<decltype(st.rbegin())>);
    assert(st.rbegin() == st.rend());
  }

  {
    std::stacktrace const st = c();
    assert(st.rbegin() != st.rend());

    auto last = st[st.size() - 1];
    auto it   = st.rbegin();
    assert(*it == last);
  }

  return 0;
}
