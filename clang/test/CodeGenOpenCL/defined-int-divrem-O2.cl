// OpenCL C v3.0 s6.5.1 gives integer division by zero an unspecified value,
// so the optimizer must not treat it as undefined behavior and delete code
// that a conforming kernel runs.

// RUN: %clang_cc1 -triple x86_64-unknown-unknown -cl-std=CL3.0 -O2 -Wno-division-by-zero -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -cl-std=CL3.0 -O2 -Wno-division-by-zero -emit-llvm -o - %s | FileCheck %s

// Without the defined lowering, a / b lets the optimizer assume b != 0 and
// remove the store of 7.
// CHECK-LABEL: define {{.*}} void @divisor_checked_after(
// CHECK:       store i32 7,
// CHECK:       ret void
__kernel void divisor_checked_after(__global int *out, int a, int b) {
  int q = a / b;
  if (b == 0)
    out[1] = 7;
  out[0] = q;
}

// Without the defined lowering, 5 / 0 is undefined and the store to out[1] is
// removed. Whatever value the division gives, out[1] is 33 or 44.
// CHECK-LABEL: define {{.*}} void @constant_zero_divisor(
// CHECK:       store i32 11, ptr{{.*}} %out
// CHECK:       [[P1:%.*]] = getelementptr inbounds nuw i8, ptr{{.*}} %out, i64 4
// CHECK:       store i32 {{.*}}, ptr{{.*}} [[P1]]
// CHECK:       [[P2:%.*]] = getelementptr inbounds nuw i8, ptr{{.*}} %out, i64 8
// CHECK:       store i32 22, ptr{{.*}} [[P2]]
// CHECK:       ret void
__kernel void constant_zero_divisor(__global int *out) {
  out[0] = 11;
  int v = 5 / 0;
  out[1] = v == 0 ? 33 : 44;
  out[2] = 22;
}
