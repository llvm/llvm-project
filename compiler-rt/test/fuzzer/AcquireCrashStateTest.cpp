// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

// Ensures that error reports are suppressed after
// __sanitizer_acquire_crash_state() has been called the first time.
#include "sanitizer/common_interface_defs.h"

#include <cassert>
#include <cstdint>
#include <cstdlib>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  assert(Data);
  if (Size == 0) return 0;
  __sanitizer_acquire_crash_state();
  exit(0);  // No report should be generated here.
}

