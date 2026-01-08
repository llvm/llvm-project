// Check that we can patch and un-patch on demand, and that logging gets invoked
// appropriately.
//
// Do not run on powerpc64le, as linking XRay with C compiler causes linker error
// due to std::__throw_system_error(int) being present in XRay libraries.
// See https://github.com/llvm/llvm-project/issues/141598
//
// RUN: %clang_xray -fxray-instrument -std=c23 %s -o %t
// RUN: env XRAY_OPTIONS="patch_premain=false" %run %t 2>&1 | FileCheck %s
// RUN: %clang_xray -fxray-instrument -fno-xray-function-index -std=c23 %s -o %t
// RUN: env XRAY_OPTIONS="patch_premain=false" %run %t 2>&1 | FileCheck %s

// UNSUPPORTED: target-is-mips64,target-is-mips64el
// UNSUPPORTED: target=powerpc64le-{{.*}}

#include "xray/xray_interface.h"

#include <stdio.h>

bool called = false;

void test_handler(int32_t fid, enum XRayEntryType type) {
  printf("called: %d, type=%d\n", fid, (int32_t)(type));
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
  printf("patching status: %d\n", (int32_t)status);
  // CHECK-NEXT: patching status: 1
  always_instrument();
  // CHECK-NEXT: called: {{.*}}, type=0
  // CHECK-NEXT: always instrumented called
  // CHECK-NEXT: called: {{.*}}, type=1
  status = __xray_unpatch();
  printf("patching status: %d\n", (int32_t)status);
  // CHECK-NEXT: patching status: 1
  always_instrument();
  // CHECK-NEXT: always instrumented called
  status = __xray_patch();
  printf("patching status: %d\n", (int32_t)status);
  // CHECK-NEXT: patching status: 1
  __xray_remove_handler();
  always_instrument();
  // CHECK-NEXT: always instrumented called
  status = __xray_unpatch();
  printf("patching status: %d\n", (int32_t)status);
  // CHECK-NEXT: patching status: 1
}
