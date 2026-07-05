// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Simple test for a fuzzer. The fuzzer must find the empty string.
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>

static volatile int *Null = 0;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (Size == 0) {
    std::cout << "Found the target, dereferencing NULL\n";
    *Null = 1;
  }
  return 0;
}

