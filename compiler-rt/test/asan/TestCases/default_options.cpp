// RUN: %clangxx_asan -O2 %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

#include "defines.h"

const char *kAsanDefaultOptions = "verbosity=1 help=1";
// Required for dyld macOS 12.0+
#if (__APPLE__)
__attribute__((weak))
#endif
ATTRIBUTE_NO_SANITIZE_ADDRESS extern "C" const char *
__asan_default_options() {
  // CHECK: Available flags for AddressSanitizer:
  return kAsanDefaultOptions;
}

int main() {
  return 0;
}
