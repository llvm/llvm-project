// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Simple test for a fuzzer. The fuzzer must find a particular string.
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  const char kString64Bytes[] =
      "123456789 123456789 123456789 123456789 123456789 123456789 1234";
  assert(sizeof(kString64Bytes) == 65);
  if (Size >= 64 && memcmp(Data, kString64Bytes, 64) == 0) {
    fprintf(stderr, "BINGO\n");
    exit(1);
  }
  return 0;
}
