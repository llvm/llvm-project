// Check that we can patch and un-patch on demand, and that logging gets invoked
// appropriately.
//
// RUN: %clangxx_xray -fxray-instrument -std=c++11 %s -o %t
// RUN: env XRAY_OPTIONS="patch_premain=false" %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_xray -fxray-instrument -fno-xray-function-index -std=c++11 %s -o %t
// RUN: env XRAY_OPTIONS="patch_premain=false" %run %t 2>&1 | FileCheck %s

// UNSUPPORTED: target-is-mips64,target-is-mips64el

#include "xray/xray_interface.h"

#include <cstdio>

bool called = false;

void test_handler(int32_t fid, XRayEntryType type) {
  printf("called: %d, type=%d\n", fid, static_cast<int32_t>(type));
  called = true;
}

[[clang::xray_always_instrument]] void always_instrument() {
  printf("always instrumented called\n");
}

int main() {
  __xray_set_handler(test_handler);
  always_instrument();
  // CHECK: always instrumented called
  auto status = __xray_patch();
  printf("patching status: %d\n", static_cast<int32_t>(status));
  // CHECK-NEXT: patching status: 1
  always_instrument();
  // CHECK-NEXT: called: {{.*}}, type=0
  // CHECK-NEXT: always instrumented called
  // CHECK-NEXT: called: {{.*}}, type=1
  status = __xray_unpatch();
  printf("patching status: %d\n", static_cast<int32_t>(status));
  // CHECK-NEXT: patching status: 1
  always_instrument();
  // CHECK-NEXT: always instrumented called
  status = __xray_patch();
  printf("patching status: %d\n", static_cast<int32_t>(status));
  // CHECK-NEXT: patching status: 1
  __xray_remove_handler();
  always_instrument();
  // CHECK-NEXT: always instrumented called
  status = __xray_unpatch();
  printf("patching status: %d\n", static_cast<int32_t>(status));
  // CHECK-NEXT: patching status: 1
}
