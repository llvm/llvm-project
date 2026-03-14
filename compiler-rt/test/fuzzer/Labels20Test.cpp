// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Simple test for a fuzzer.
// Needs to find a string "FUZZxxxxxxxxxxxxMxxE", where 'x' is any byte.
#include <assert.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>

extern "C" bool Func1(const uint8_t *Data, size_t Size);
extern "C" bool Func2(const uint8_t *Data, size_t Size);

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (Size >= 20
      && Data[0] == 'F'
      && Data[1] == 'U'
      && Data[2] == 'Z'
      && Data[3] == 'Z'
      && Func1(Data, Size)
      && Func2(Data, Size)) {
        fprintf(stderr, "BINGO\n");
        abort();
  }
  return 0;
}

extern "C"
__attribute__((noinline))
bool Func1(const uint8_t *Data, size_t Size) {
  // assumes Size >= 5, doesn't check it.
  return Data[16] == 'M';
}

extern "C"
__attribute__((noinline))
bool Func2(const uint8_t *Data, size_t Size) {
  return Size >= 20 && Data[19] == 'E';
}
