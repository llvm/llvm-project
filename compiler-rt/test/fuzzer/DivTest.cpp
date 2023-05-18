// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Simple test for a fuzzer: find the interesting argument for div.
#include <assert.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>

static volatile int Sink;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (Size < 4) return 0;
  int a;
  memcpy(&a, Data, 4);
  Sink = 12345678 / (987654 - a);
  return 0;
}

