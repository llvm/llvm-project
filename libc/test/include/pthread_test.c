//===-- Unittests for pthread macro ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "include/llvm-libc-macros/pthread-macros.h"
#include "include/llvm-libc-types/pthread_t.h"

#include <assert.h>

pthread_t p = PTHREAD_NULL;

int main(void) {
  assert(__PTHREAD_GET_ID(p) == 0);
  return 0;
}
