// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

// Test that we don't creash in case of bad strcmp params.
#include <cstddef>
#include <cstdint>
#include <cstring>

static volatile int Sink;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (Size != 10) return 0;
  // Data is not zero-terminated, so this call is bad.
  // Still, there are cases when such calles appear, see e.g.
  // https://bugs.llvm.org/show_bug.cgi?id=32357
  Sink = strcmp(reinterpret_cast<const char*>(Data), "123456789");
  return 0;
}

