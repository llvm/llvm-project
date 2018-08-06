// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

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
