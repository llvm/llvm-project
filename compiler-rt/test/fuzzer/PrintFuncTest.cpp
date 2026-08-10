// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Simple test for a fuzzer. The fuzzer must find the string "Hi!".
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>

extern "C" {
__attribute__((noinline))
void FunctionC(const uint8_t *Data, size_t Size) {
  if (Size > 3 && Data[3] == 'Z') {
    static bool PrintedOnce = false;
    if (!PrintedOnce) {
      std::cout << "BINGO\n";
      PrintedOnce = true;
    }
  }
}

__attribute__((noinline))
void FunctionB(const uint8_t *Data, size_t Size) {
  if (Size > 2 && Data[2] == 'Z')
    FunctionC(Data, Size);
}
__attribute__((noinline))
void FunctionA(const uint8_t *Data, size_t Size) {
  if (Size > 1 && Data[1] == 'U')
    FunctionB(Data, Size);
}
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (Size > 0 && Data[0] == 'F')
    FunctionA(Data, Size);
  return 0;
}

