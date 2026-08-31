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
//   const_reference operator[](size_type) const;

#include <cassert>
#include <stacktrace>
#include "test_macros.h"

// Call chain is: main -> c -> b -> a -> stacktrace::current
TEST_NO_TAIL_CALLS TEST_NOINLINE std::stacktrace a() { return std::stacktrace::current(); }
TEST_NO_TAIL_CALLS TEST_NOINLINE std::stacktrace b() { return a(); }
TEST_NO_TAIL_CALLS TEST_NOINLINE std::stacktrace c() { return b(); }

int main(int, char**) {
  auto const st = c();
  assert(st[0]);
  assert(st[1]);
  assert(st[2]);
  assert(st[3]);

  auto f0 = st[0];
  auto f1 = st[1];
  auto f2 = st[2];
  auto f3 = st[3];

  auto it = st.begin();
  assert(*it++ == f0);
  assert(it != st.end());
  assert(*it++ == f1);
  assert(it != st.end());
  assert(*it++ == f2);
  assert(it != st.end());
  assert(*it++ == f3);

  return 0;
}
