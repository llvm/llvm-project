// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Find FUZ
#include <cstddef>
#include <cstdint>
#include <cstdlib>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (Size < 3) return 0;
  uint32_t x = Data[0] + 251 * Data[1] + 251 * 251 * Data[2];
  if (x == 'F' + 251 * 'U' + 251 * 251 * 'Z')     abort();
  return 0;
}
