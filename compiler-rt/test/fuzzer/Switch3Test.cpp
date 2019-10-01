// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Simple test for a fuzzer. The fuzzer must find the interesting switch value.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static volatile int Sink;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  uint32_t v;
  if (Size < 100) return 0;
  memcpy(&v, Data + Size / 2, sizeof(v));
  switch(v) {
    case 0x47524159: abort();
    case 0x52474220: abort();
    default:;
  }
  return 0;
}

