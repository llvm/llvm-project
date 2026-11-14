// REQUIRES: x86-registered-target
// Clang emits TBAA only when optimizing, so this test needs -O1.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

// A struct or _Complex output with a register constraint is returned as an
// integer and stored to the destination. The store must not have the integer's
// TBAA tag, or it wouldn't alias accesses to the destination's members.

struct S {
  float a, b;
};

// CHECK-LABEL: define {{.*}}void @test_struct_output(
// CHECK:         [[RES:%.*]] = call i64 asm "", "=r,
// CHECK-NEXT:    store i64 [[RES]], ptr %{{.*}}, align 4, !tbaa [[CHAR_TAG:![0-9]+]]
void test_struct_output(struct S *p) {
  asm("" : "=r"(*p));
}

// CHECK-LABEL: define {{.*}}void @test_complex_output(
// CHECK:         [[RES:%.*]] = call i64 asm "", "=r,
// CHECK-NEXT:    store i64 [[RES]], ptr %{{.*}}, align 4, !tbaa [[CHAR_TAG]]
void test_complex_output(_Complex float *p) {
  asm("" : "=r"(*p));
}

// CHECK-DAG: [[CHAR_TAG]] = !{[[CHAR:![0-9]+]], [[CHAR]], i64 0}
// CHECK-DAG: [[CHAR]] = !{!"omnipotent char",
