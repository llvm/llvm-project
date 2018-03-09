// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

// Check that allocation tracing from different threads does not cause
// interleaving of stack traces.
#include <assert.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <thread>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  auto C = [&] {
    volatile void *a = malloc(5639);
    free((void *)a);
  };
  std::thread T[] = {std::thread(C), std::thread(C), std::thread(C),
                     std::thread(C), std::thread(C), std::thread(C)};
  for (auto &X : T)
    X.join();
  return 0;
}
