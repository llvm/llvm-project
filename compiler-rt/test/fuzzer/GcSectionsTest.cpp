// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

// Simple test for a fuzzer.
// The unused function should not be present in the binary.
#include <cstddef>
#include <cstdint>

extern "C" void UnusedFunctionShouldBeRemovedByLinker() { }

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  return 0;
}

