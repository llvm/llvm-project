// Verify that AArch64 XRay trampoline symbols are present and functional.

// RUN: %clangxx_xray -fxray-instruction-threshold=1 %s -o %t
// RUN: nm %t | FileCheck %s --check-prefix SYMBOLS
// RUN: %run %t 2>&1 | FileCheck %s

// REQUIRES: target=arm64-apple-{{.*}}

// SYMBOLS: __xray_FunctionEntry
// SYMBOLS: __xray_FunctionExit

#include "xray/xray_interface.h"
#include <cstdio>

static int entries = 0;
static int exits = 0;

[[clang::xray_never_instrument]] void handler(int32_t fid, XRayEntryType type) {
  if (type == XRayEntryType::ENTRY)
    ++entries;
  else if (type == XRayEntryType::EXIT)
    ++exits;
}

[[clang::xray_always_instrument]] int add(int a, int b) { return a + b; }

int main() {
  __xray_set_handler(handler);
  __xray_patch();
  int r = add(3, 4);
  __xray_unpatch();
  printf("entries=%d exits=%d result=%d\n", entries, exits, r);
  return 0;
}

// CHECK: entries=1 exits=1 result=7
