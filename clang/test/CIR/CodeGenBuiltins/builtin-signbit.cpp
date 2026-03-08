// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: FileCheck --input-file=%t.cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

int test_signbit_float(float x) {
  return __builtin_signbit(x);
}
// CIR: cir.signbit %{{.*}} : !cir.float -> !cir.bool
// LLVM: bitcast float {{.*}} to i32
// LLVM: icmp slt i32 {{.*}}, 0
// LLVM: zext i1 {{.*}} to i32
// LLVM: ret i32

int test_signbit_double(double x) {
  return __builtin_signbit(x);
}
// CIR: cir.signbit %{{.*}} : !cir.double -> !cir.bool
// LLVM: bitcast double {{.*}} to i64
// LLVM: icmp slt i64 {{.*}}, 0
// LLVM: zext i1 {{.*}} to i32
// LLVM: ret i32

int test_signbit_long_double(long double x) {
  return __builtin_signbit(x);
}
// CIR: cir.signbit %{{.*}} : !cir.long_double<!cir.f80> -> !cir.bool
// LLVM: bitcast x86_fp80 {{.*}} to i80
// LLVM: icmp slt i80 {{.*}}, 0
// LLVM: zext i1 {{.*}} to i32
// LLVM: ret i32

int test_signbitf(float x) {
  return __builtin_signbitf(x);
}
// CIR: cir.signbit %{{.*}} : !cir.float -> !cir.bool
// LLVM: bitcast float {{.*}} to i32
// LLVM: icmp slt i32 {{.*}}, 0
// LLVM: zext i1 {{.*}} to i32
// LLVM: ret i32

int test_signbitl(long double x) {
  return __builtin_signbitl(x);
}
// CIR: cir.signbit %{{.*}} : !cir.long_double<!cir.f80> -> !cir.bool
// LLVM: bitcast x86_fp80 {{.*}} to i80
// LLVM: icmp slt i80 {{.*}}, 0
// LLVM: zext i1 {{.*}} to i32
// LLVM: ret i32
