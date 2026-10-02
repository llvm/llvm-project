// Verify that XRay runtime code patching works on macOS (mprotect path).
// Tests that sleds can be toggled between patched and unpatched state.

// RUN: %clangxx_xray -fxray-instruction-threshold=1 %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

// REQUIRES: target={{(arm64|x86_64)-apple-.*}}

#include "xray/xray_interface.h"
#include <cstdio>

static int call_count = 0;

[[clang::xray_never_instrument]] void handler(int32_t fid, XRayEntryType type) {
  if (type == XRayEntryType::ENTRY)
    ++call_count;
}

[[clang::xray_always_instrument]] int target_fn(int x) { return x + 1; }

int main() {
  // Before patching: handler should not fire.
  target_fn(1);
  printf("before_patch: %d\n", call_count);

  // Patch and call: handler should fire.
  __xray_set_handler(handler);
  __xray_patch();
  target_fn(2);
  printf("after_patch: %d\n", call_count);

  // Unpatch and call: handler should stop firing.
  __xray_unpatch();
  target_fn(3);
  printf("after_unpatch: %d\n", call_count);

  // Re-patch: handler fires again.
  __xray_patch();
  target_fn(4);
  printf("after_repatch: %d\n", call_count);

  return 0;
}

// CHECK: before_patch: 0
// CHECK: after_patch: 1
// CHECK: after_unpatch: 1
// CHECK: after_repatch: 2
