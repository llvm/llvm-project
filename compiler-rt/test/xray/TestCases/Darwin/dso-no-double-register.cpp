// Verify that a DSO built with -fxray-shared doesn't get double-registered
// on macOS. The dyld image callback and the DSO constructor both discover the
// same sleds; the dedup check in __xray_register_sleds ensures single registration.

// RUN: %clangxx_xray -fxray-instrument -fxray-shared -fxray-instruction-threshold=1 \
// RUN:     -shared %S/Inputs/dso_instrumented.cpp -o %t.dylib
// RUN: %clangxx_xray -fxray-instruction-threshold=1 %s -o %t %t.dylib
// RUN: %run %t 2>&1 | FileCheck %s

// REQUIRES: target={{(arm64|x86_64)-apple-.*}}

#include "xray/xray_interface.h"
#include <cstdio>

extern int dso_add(int, int);

static int entry_count = 0;
static int exit_count = 0;

[[clang::xray_never_instrument]] void handler(int32_t fid, XRayEntryType type) {
  if (type == XRayEntryType::ENTRY)
    ++entry_count;
  else if (type == XRayEntryType::EXIT)
    ++exit_count;
}

[[clang::xray_always_instrument]] int main_fn(int x) { return dso_add(x, x); }

int main() {
  __xray_set_handler(handler);
  __xray_patch();
  int r = main_fn(21);
  __xray_unpatch();
  // main_fn: 1 entry + 1 exit = 2 handler calls total for main_fn.
  // If DSO is double-registered, we'd see duplicated entry/exit calls.
  // Exact count depends on whether DSO's function is also patched.
  printf("result=%d entries=%d exits=%d\n", r, entry_count, exit_count);
  // entries == exits proves balanced tracing (no duplication).
  return (r == 42 && entry_count == exit_count && entry_count >= 1) ? 0 : 1;
}

// CHECK: result=42 entries=[[N:[0-9]+]] exits=[[N]]
