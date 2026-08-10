// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests -trace_malloc
#include <assert.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>

int *Ptr;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (!Size) return 0;
  if (*Data == 1) {
    delete Ptr;
    Ptr = nullptr;
  } else if (*Data == 2) {
    delete Ptr;
    Ptr = new int;
  } else if (*Data == 3) {
    if (!Ptr)
      Ptr = new int;
  }
  return 0;
}

