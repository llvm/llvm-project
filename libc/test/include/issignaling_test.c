//===-- Unittests for issignaling macro -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDSList-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "include/llvm-libc-macros/math-function-macros.h"

#include <assert.h>

// TODO: enable the test unconditionally when issignaling macro is fixed for
//       older compiler
#if (defined(__clang__) && __clang_major__ >= 18) ||                           \
    (defined(__GNUC__) && __GNUC__ >= 13)

// check if macro is defined
#ifndef issignaling
#error "issignaling macro is not defined"
#else
int main(void) {
  assert(issignaling(__builtin_nans("")) == 1);
  assert(issignaling(__builtin_nansf("")) == 1);
  assert(issignaling(__builtin_nansl("")) == 1);
  assert(issignaling(1.819f) == 0);
  assert(issignaling(-1.726) == 0);
  assert(issignaling(1.426L) == 0);
  return 0;
}
#endif

#endif
