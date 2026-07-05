// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Simple test for a fuzzer. This test may trigger two different bugs.
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>

static volatile int *Null = 0;

void Foo() { Null[1] = 0; }
void Bar() { Null[2] = 0; }

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (Size < 10 && Data[0] == 'H')
    Foo();
  if (Size >= 10 && Data[0] == 'H')
    Bar();
  return 0;
}

