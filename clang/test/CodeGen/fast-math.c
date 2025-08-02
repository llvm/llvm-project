// RUN: %clang_cc1 -ffast-math -ffp-contract=fast -emit-llvm -O2 -o - %s | FileCheck %s
float f0, f1, f2;

void foo(void) {
  // CHECK-LABEL: define {{.*}}void @foo()

  // CHECK: fadd fast
  f0 = f1 + f2;

  // CHECK: ret
}

float issue_84648a(float *x) {
  return x[0] == x[1] ? x[1] : x[0];
}

// CHECK-LABEL: define{{.*}} float @issue_84648a(ptr {{.*}})
// CHECK:       [[VAL:%.+]] = load float, ptr
// CHECK:       ret float [[VAL]]

float issue_84648b(float *x) {
#pragma float_control(precise, on)
  return x[0] == x[1] ? x[1] : x[0];
}

// CHECK-LABEL: define{{.*}} float @issue_84648b(ptr{{.*}} %x)
// CHECK:       fcmp oeq
