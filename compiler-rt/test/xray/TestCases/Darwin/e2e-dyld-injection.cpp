// End-to-end test for DYLD_INSERT_LIBRARIES injection.
// Compiles a binary with XRay instrumentation but without the runtime linked,
// then injects the XRay runtime dylib to discover and patch the sleds.

// Build a dylib containing the XRay runtime:
// RUN: %clang -shared -o %t.xray-rt.dylib \
// RUN:   -Wl,-force_load,%xray_rt_path \
// RUN:   -lc++

// Build target binary with XRay sleds but without XRay runtime:
// RUN: %clangxx -fxray-instrument -fxray-instruction-threshold=1 -fno-xray-link-deps \
// RUN:   %s -o %t.target

// Inject and run — XRay should discover sleds in the host binary:
// RUN: env DYLD_INSERT_LIBRARIES=%t.xray-rt.dylib \
// RUN:   XRAY_OPTIONS="patch_premain=true verbosity=1" \
// RUN:   %run %t.target 2>&1 | FileCheck %s

// REQUIRES: target={{(arm64|x86_64)-apple-.*}}

// CHECK: Registering {{[0-9]+}} new functions
// CHECK: Patching object
// CHECK: sum=20

#include <cstdio>

[[clang::xray_always_instrument]] int target_fn(int x) { return x * 2; }

int main() {
  int sum = 0;
  for (int i = 0; i < 5; ++i)
    sum += target_fn(i);
  printf("sum=%d\n", sum);
  return 0;
}
