// Check that we can patch and un-patch on demand, and that logging gets invoked
// appropriately.
//

// RUN: split-file %s %t
// RUN: %clangxx_xray -g -fPIC -fxray-instrument -fxray-shared -shared -std=c++11 %t/testlib.cpp -o %t/testlib.so
// RUN: %clangxx_xray -g -fPIC -fxray-instrument -fxray-shared -std=c++11 %t/main.cpp %t/testlib.so -Wl,-rpath,%t -o %t/main.o

// RUN: XRAY_OPTIONS="patch_premain=false" %run %t/main.o 2>&1 | FileCheck %s

// REQUIRES: target={{(aarch64|x86_64)-.*}}

//--- main.cpp

#include "xray/xray_interface.h"

#include <cstdio>

bool called = false;

void test_handler(int32_t fid, XRayEntryType type) {
  printf("called: %d, type=%d\n", fid, static_cast<int32_t>(type));
  called = true;
}

[[clang::xray_always_instrument]] void instrumented_in_executable() {
  printf("instrumented_in_executable called\n");
}

extern void instrumented_in_dso();

int main() {
  __xray_set_handler(test_handler);
  instrumented_in_executable();
  // CHECK: instrumented_in_executable called
  instrumented_in_dso();
  // CHECK: instrumented_in_dso called
  auto status = __xray_patch();
  printf("patching status: %d\n", static_cast<int32_t>(status));
  // CHECK-NEXT: patching status: 1
  instrumented_in_executable();
  // CHECK-NEXT: called: {{.*}}, type=0
  // CHECK-NEXT: instrumented_in_executable called
  // CHECK-NEXT: called: {{.*}}, type=1
  instrumented_in_dso();
  // CHECK-NEXT: called: {{.*}}, type=0
  // CHECK-NEXT: instrumented_in_dso called
  // CHECK-NEXT: called: {{.*}}, type=1
  status = __xray_unpatch();
  printf("patching status: %d\n", static_cast<int32_t>(status));
  // CHECK-NEXT: patching status: 1
  instrumented_in_executable();
  // CHECK-NEXT: instrumented_in_executable called
  instrumented_in_dso();
  // CHECK-NEXT: instrumented_in_dso called
  status = __xray_patch();
  printf("patching status: %d\n", static_cast<int32_t>(status));
  // CHECK-NEXT: patching status: 1
  __xray_remove_handler();
  instrumented_in_executable();
  // CHECK-NEXT: instrumented_in_executable called
  instrumented_in_dso();
  // CHECK-NEXT: instrumented_in_dso called
  status = __xray_unpatch();
  printf("patching status: %d\n", static_cast<int32_t>(status));
  // CHECK-NEXT: patching status: 1
}

//--- testlib.cpp

#include <cstdio>

[[clang::xray_always_instrument]] void instrumented_in_dso() {
  printf("instrumented_in_dso called\n");
}
