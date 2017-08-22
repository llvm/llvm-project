// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

// This test should not be instrumented.
#include <cstddef>
#include <cstdint>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  return 0;
}

