// C keeps undefined behavior for integer division and remainder by default;
// -fdefined-int-divrem gives them an unspecified value instead.

// RUN: %clang_cc1 -disable-llvm-passes -triple x86_64-unknown-unknown -fenable-matrix -emit-llvm -o - %s | FileCheck %s --check-prefix=PLAIN --implicit-check-not=.poison.
// RUN: %clang_cc1 -disable-llvm-passes -triple x86_64-unknown-unknown -fenable-matrix -fdefined-int-divrem -emit-llvm -o - %s | FileCheck %s --check-prefix=POISON

int f_div(int a, int b) { return a / b; }
// PLAIN-LABEL: define {{.*}} @f_div
// PLAIN: = sdiv i32
// POISON-LABEL: define {{.*}} @f_div
// POISON: = call i32 @llvm.sdiv.poison.i32(i32
// POISON-NEXT: freeze i32

int f_rem(int a, int b) { return a % b; }
// PLAIN-LABEL: define {{.*}} @f_rem
// PLAIN: = srem i32
// POISON-LABEL: define {{.*}} @f_rem
// POISON: = call i32 @llvm.srem.poison.i32(i32
// POISON-NEXT: freeze i32

// A matrix is divided by a splat of the scalar.
typedef int m2x2 __attribute__((matrix_type(2, 2)));
m2x2 f_matdiv(m2x2 m, int b) { return m / b; }
// PLAIN-LABEL: define {{.*}} @f_matdiv
// PLAIN: = sdiv <4 x i32>
// POISON-LABEL: define {{.*}} @f_matdiv
// POISON: [[SPLAT:%.*]] = shufflevector <4 x i32>
// POISON: = call <4 x i32> @llvm.sdiv.poison.v4i32(<4 x i32> {{.*}}, <4 x i32> [[SPLAT]])
// POISON-NEXT: freeze <4 x i32>
