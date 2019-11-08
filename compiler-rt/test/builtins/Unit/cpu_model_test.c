// FIXME: XFAIL the test because it is expected to return non-zero value.
// XFAIL: *
// REQUIRES: x86-target-arch
// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_cpu_model
//===-- cpu_model_test.c - Test __builtin_cpu_supports --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file tests __builtin_cpu_supports for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>

int main (void) {
#if defined(i386) || defined(__x86_64__)
  if(__builtin_cpu_supports("avx2"))
    return 4;
  else
    return 3;
#else
  printf("skipped\n");
  return 0;
#endif
}
