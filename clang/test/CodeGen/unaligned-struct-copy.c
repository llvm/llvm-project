// RUN: %clang_cc1 -xc   -O2 -triple thumbv7a-unknown-windows-eabi -fms-extensions -emit-llvm < %s | FileCheck %s
// RUN: %clang_cc1 -xc++ -O2 -triple thumbv7a-unknown-windows-eabi -fms-extensions -emit-llvm < %s | FileCheck %s
// RUN: %clang_cc1 -xc   -O2 -triple x86_64-unknown-linux-gnu -fms-extensions -emit-llvm < %s | FileCheck %s
// RUN: %clang_cc1 -xc++ -O2 -triple x86_64-unknown-linux-gnu -fms-extensions -emit-llvm < %s | FileCheck %s

struct S1 {
  unsigned long x;
};

// CHECK: define
// CHECK-SAME: void
// CHECK-SAME: test1

void test1(__unaligned struct S1 *out) {
  // CHECK: store
  // CHECK-SAME: align 1
  out->x = 5;
  // CHECK: ret void
}

// CHECK: define
// CHECK-SAME: void
// CHECK-SAME: test2

void test2(__unaligned struct S1 *out, __unaligned struct S1 *in) {
  // CHECK: load
  // CHECK-SAME: align 1
  // CHECK: store
  // CHECK-SAME: align 1
  *out = *in;
  // CHECK: ret void
}
