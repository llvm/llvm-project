// Checking that DSOs are automatically patched upon load, if patch_premain is passed.

// RUN: split-file %s %t
// RUN: %clangxx_xray -g -fPIC -fxray-instrument -fxray-shared -shared -std=c++11 %t/testlib.cpp -o %t/testlib.so
// RUN: %clangxx_xray -g -fPIC -fxray-instrument -fxray-shared -std=c++11 %t/main.cpp %t/testlib.so -Wl,-rpath,%t -o %t/main.o

// RUN: XRAY_OPTIONS="patch_premain=true,verbosity=1" %run %t/main.o 2>&1 | FileCheck %s

// REQUIRES: target={{(aarch64|x86_64)-.*}}

//--- main.cpp

#include "xray/xray_interface.h"

#include <cstdio>

void test_handler(int32_t fid, XRayEntryType type) {
  printf("called: %d, type=%d\n", fid, static_cast<int32_t>(type));
}

[[clang::xray_always_instrument]] void instrumented_in_executable() {
  printf("instrumented_in_executable called\n");
}

extern void instrumented_in_dso();

int main() {
  __xray_set_handler(test_handler);
  instrumented_in_executable();
  // CHECK: called: {{.*}}, type=0
  // CHECK-NEXT: instrumented_in_executable called
  // CHECK-NEXT: called: {{.*}}, type=1
  instrumented_in_dso();
  // CHECK-NEXT: called: {{.*}}, type=0
  // CHECK-NEXT: instrumented_in_dso called
  // CHECK-NEXT: called: {{.*}}, type=1
}

//--- testlib.cpp

#include <cstdio>

[[clang::xray_always_instrument]] void instrumented_in_dso() {
  printf("instrumented_in_dso called\n");
}
