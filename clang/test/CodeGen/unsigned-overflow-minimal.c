// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsanitize=unsigned-integer-overflow -fsanitize-minimal-runtime %s -emit-llvm -o - | FileCheck %s

unsigned long li, lj, lk;

// CHECK-LABEL: define{{.*}} void @testlongadd()
void testlongadd() {
  // CHECK: call void @__ubsan_handle_add_overflow_minimal_abort()
  li = lj + lk;
}

// CHECK-LABEL: define{{.*}} void @testlongsub()
void testlongsub() {
  // CHECK: call void @__ubsan_handle_sub_overflow_minimal_abort()
  li = lj - lk;
}

// CHECK-LABEL: define{{.*}} void @testlongmul()
void testlongmul() {
  // CHECK: call void @__ubsan_handle_mul_overflow_minimal_abort()
  li = lj * lk;
}
