// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test with a more mallocs than frees, but no leak.
#include <cstddef>
#include <cstdint>

const int kAllocatedPointersSize = 10000;
int NumAllocatedPointers = 0;
int *AllocatedPointers[kAllocatedPointersSize];

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (NumAllocatedPointers < kAllocatedPointersSize)
    AllocatedPointers[NumAllocatedPointers++] = new int;
  return 0;
}

