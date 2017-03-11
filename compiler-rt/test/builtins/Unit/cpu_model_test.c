// FIXME: XFAIL the test because it is expected to return non-zero value.
// XFAIL: *
// REQUIRES: x86-target-arch
// RUN: %clang_builtins %s %librt -o %t && %run %t
//===-- cpu_model_test.c - Test __builtin_cpu_supports --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
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
