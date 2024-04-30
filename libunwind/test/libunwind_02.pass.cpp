// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// TODO: Figure out why this fails with Memory Sanitizer.
// XFAIL: msan

// This test fails on older llvm, when built with picolibc.
// XFAIL: clang-16 && LIBCXX-PICOLIBC-FIXME

#undef NDEBUG
#include <assert.h>
#include <stdlib.h>
#include <unwind.h>

#define EXPECTED_NUM_FRAMES 50
#define NUM_FRAMES_UPPER_BOUND 100

__attribute__((noinline)) _Unwind_Reason_Code callback(_Unwind_Context *context,
                                                       void *cnt) {
  (void)context;
  int *i = (int *)cnt;
  ++*i;
  if (*i > NUM_FRAMES_UPPER_BOUND) {
    abort();
  }
  return _URC_NO_REASON;
}

__attribute__((noinline)) void test_backtrace() {
  int n = 0;
  _Unwind_Backtrace(&callback, &n);
  if (n < EXPECTED_NUM_FRAMES) {
    abort();
  }
}

// These functions are effectively the same, but we have to be careful to avoid
// unwanted optimizations that would mess with the number of frames we expect.
// Surprisingly, slapping `noinline` is not sufficient -- we also have to avoid
// writing the function in a way that the compiler can easily spot tail
// recursion.
__attribute__((noinline)) int test1(int i);
__attribute__((noinline)) int test2(int i);

__attribute__((noinline)) int test1(int i) {
  if (i == 0) {
    test_backtrace();
    return 0;
  } else {
    return i + test2(i - 1);
  }
}

__attribute__((noinline)) int test2(int i) {
  if (i == 0) {
    test_backtrace();
    return 0;
  } else {
    return i + test1(i - 1);
  }
}

int main(int, char**) {
  int total = test1(50);
  assert(total == 1275);
  return 0;
}
