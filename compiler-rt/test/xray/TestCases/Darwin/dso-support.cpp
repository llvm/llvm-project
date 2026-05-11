// Verify that XRay supports DSO (shared library) tracing on macOS.
// The DSO-local runtime registers its sleds with the main runtime.

// RUN: %clangxx_xray -fxray-instrument -fxray-shared -fxray-instruction-threshold=1 \
// RUN:     -shared %S/Inputs/dso_instrumented.cpp -o %t.dylib
// RUN: %clangxx_xray -fxray-instruction-threshold=1 %s -o %t %t.dylib
// RUN: %run %t 2>&1 | FileCheck %s

// REQUIRES: target={{(arm64|x86_64)-apple-.*}}

#include "xray/xray_interface.h"
#include <cstdio>

extern int dso_add(int, int);

static int entry_count = 0;

[[clang::xray_never_instrument]] void handler(int32_t fid, XRayEntryType type) {
  if (type == XRayEntryType::ENTRY)
    ++entry_count;
}

[[clang::xray_always_instrument]] int main_fn(int x) { return dso_add(x, x); }

int main() {
  __xray_set_handler(handler);
  __xray_patch();
  int r = main_fn(21);
  __xray_unpatch();
  // main_fn entry = at least 1 entry
  printf("result=%d entries=%d\n", r, entry_count);
  return r == 42 && entry_count >= 1 ? 0 : 1;
}

// CHECK: result=42 entries={{[1-9][0-9]*}}
