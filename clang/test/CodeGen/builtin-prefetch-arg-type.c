// RUN: %clang_cc1 -triple x86_64-pc-linux -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple msp430 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple avr -emit-llvm %s -o - | FileCheck %s

// The hint operands of llvm.prefetch are i32 regardless of the type of the
// corresponding C argument, which is narrower on targets with a 16-bit int and
// wider when the caller passes a long.

void test_int(void *p) {
  // CHECK: call{{.*}} void @llvm.prefetch.p0(ptr {{%.+}}, i32 1, i32 3, i32 1)
  __builtin_prefetch(p, 1, 3);
}

void test_long(void *p) {
  // CHECK: call{{.*}} void @llvm.prefetch.p0(ptr {{%.+}}, i32 1, i32 3, i32 1)
  __builtin_prefetch(p, 1L, 3L);
}

void test_defaulted_hints(void *p) {
  // CHECK: call{{.*}} void @llvm.prefetch.p0(ptr {{%.+}}, i32 0, i32 3, i32 1)
  __builtin_prefetch(p);
}
