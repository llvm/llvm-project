// Verify that XRay works with pointer authentication (PAC) enabled.
// The XRay sled is placed before PACIASP in the function prologue,
// so the trampoline receives an unsigned LR, and PAC signing happens
// after the trampoline returns.

// RUN: %clangxx_xray -fxray-instruction-threshold=1 -mbranch-protection=pac-ret %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

// REQUIRES: target=arm64{{(-apple)?.*}}

#include "xray/xray_interface.h"
#include <cstdio>

static int handler_calls = 0;

[[clang::xray_never_instrument]] void handler(int32_t fid, XRayEntryType type) {
  ++handler_calls;
}

[[clang::xray_always_instrument]] int compute(int x) { return x * 2 + 1; }

[[clang::xray_always_instrument]] int nested(int x) {
  return compute(x) + compute(x + 1);
}

int main() {
  __xray_set_handler(handler);
  __xray_patch();
  int r = nested(5);
  __xray_unpatch();
  printf("result=%d handler_calls=%d\n", r, handler_calls);
  return 0;
}

// CHECK: result=24 handler_calls=6
