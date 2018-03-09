// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

// Simple test for a fuzzer: it simply sleeps for 1 second.
#include <cstddef>
#include <cstdint>
#include <thread>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  std::this_thread::sleep_for(std::chrono::seconds(1));
  return 0;
}

