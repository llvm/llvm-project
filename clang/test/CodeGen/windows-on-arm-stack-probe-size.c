// RUN: %clang_cc1 -triple thumbv7-windows-itanium -mstack-probe-size=8096 -fms-extensions -O2 -emit-llvm %s -o - \
// RUN:     | FileCheck %s -check-prefix CHECK-8096

// RUN: %clang_cc1 -triple thumbv7-windows-itanium -mstack-probe-size=4096 -fms-extensions -O2 -emit-llvm %s -o - \
// RUN:     | FileCheck %s -check-prefix CHECK-4096

// RUN: %clang_cc1 -triple thumbv7-windows-itanium -fms-extensions -O2 -emit-llvm %s -o - \
// RUN:     | FileCheck %s

__declspec(dllimport) void initialise(signed char buffer[4096]);

__declspec(dllexport) signed char function(unsigned index) {
  signed char buffer[4096];
  initialise(buffer);
  return buffer[index];
}

// CHECK-8096: attributes #0 = {
// CHECK-8096: "stack-probe-size"="8096"
// CHECK-8096: }

// CHECK-4096: attributes #0 = {
// CHECK-4096-NOT: "stack-probe-size"=
// CHECK-4096: }

// CHECK: attributes #0 = {
// CHECK-NOT: "stack-probe-size"=
// CHECK: }
