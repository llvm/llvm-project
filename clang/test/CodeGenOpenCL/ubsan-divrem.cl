// OpenCL C v3.0 s6.5.1 defines both conditions that
// __ubsan_handle_divrem_overflow diagnoses: integer division whose result lies
// outside the range of the integer type -- INT_MIN / -1 -- and integer divide
// by zero do not cause an exception but result in an unspecified value. Clang
// therefore does not emit the check in OpenCL C or C++ for OpenCL, the same way
// it does not emit -fsanitize=shift-exponent there.
//
// Overflow of +, - and * is not covered by that wording, so the arithmetic
// controls must remain instrumented whenever signed-integer-overflow is
// enabled.
//
// The KERNEL/GLOBAL macros come from the command line so that the same source
// can be compiled as OpenCL C, as C++ for OpenCL, as C and as SYCL device code.
// Every run passes -disable-llvm-passes so the checks see the frontend's own
// output rather than the output of the default OpenCL -O2 pipeline.

// OpenCL C omits divrem checks while retaining arithmetic checks.
// RUN: %clang_cc1 %s -triple spir64-unknown-unknown -emit-llvm -o - \
// RUN:   -disable-llvm-passes -DKERNEL=__kernel -DGLOBAL=__global \
// RUN:   -fsanitize=integer-divide-by-zero,signed-integer-overflow \
// RUN:   -fsanitize-minimal-runtime \
// RUN:   | FileCheck %s --check-prefixes=CHECK,ARITH \
// RUN:       --implicit-check-not=__ubsan_handle_divrem_overflow
//
// C++ for OpenCL omits divrem checks while retaining arithmetic checks.
// RUN: %clang_cc1 %s -triple spir64-unknown-unknown -cl-std=clc++2021 \
// RUN:   -emit-llvm -disable-llvm-passes -o - \
// RUN:   -DKERNEL=__kernel -DGLOBAL=__global \
// RUN:   -fsanitize=integer-divide-by-zero,signed-integer-overflow \
// RUN:   -fsanitize-minimal-runtime \
// RUN:   | FileCheck %s --check-prefixes=CHECK,ARITH \
// RUN:       --implicit-check-not=__ubsan_handle_divrem_overflow
//
// OpenCL C omits integer-divide-by-zero checks.
// RUN: %clang_cc1 %s -triple spir64-unknown-unknown -emit-llvm -o - \
// RUN:   -disable-llvm-passes -DKERNEL=__kernel -DGLOBAL=__global \
// RUN:   -fsanitize=integer-divide-by-zero -fsanitize-minimal-runtime \
// RUN:   | FileCheck %s --check-prefix=CHECK \
// RUN:       --implicit-check-not=__ubsan_handle_divrem_overflow
//
// OpenCL C omits signed-overflow divrem checks but retains arithmetic checks.
// RUN: %clang_cc1 %s -triple spir64-unknown-unknown -emit-llvm -o - \
// RUN:   -disable-llvm-passes -DKERNEL=__kernel -DGLOBAL=__global \
// RUN:   -fsanitize=signed-integer-overflow -fsanitize-minimal-runtime \
// RUN:   | FileCheck %s --check-prefixes=CHECK,ARITH \
// RUN:       --implicit-check-not=__ubsan_handle_divrem_overflow
//
// C retains both divrem and arithmetic checks.
// RUN: %clang_cc1 %s -x c -triple x86_64-unknown-linux-gnu -emit-llvm -o - \
// RUN:   -disable-llvm-passes -DKERNEL= -DGLOBAL= \
// RUN:   -fsanitize=integer-divide-by-zero,signed-integer-overflow \
// RUN:   -fsanitize-minimal-runtime \
// RUN:   | FileCheck %s --check-prefixes=CHECK,ARITH,SIGNED,UNSIGNED
//
// SYCL device C++ retains both divrem and arithmetic checks.
// RUN: %clang_cc1 %s -x c++ -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -emit-llvm -disable-llvm-passes -o - \
// RUN:   -DKERNEL='[[clang::sycl_external]]' -DGLOBAL= \
// RUN:   -fsanitize=integer-divide-by-zero,signed-integer-overflow \
// RUN:   -fsanitize-minimal-runtime \
// RUN:   | FileCheck %s --check-prefixes=CHECK,ARITH,SIGNED,UNSIGNED
//
// C retains integer-divide-by-zero checks.
// RUN: %clang_cc1 %s -x c -triple x86_64-unknown-linux-gnu -emit-llvm -o - \
// RUN:   -disable-llvm-passes -DKERNEL= -DGLOBAL= \
// RUN:   -fsanitize=integer-divide-by-zero -fsanitize-minimal-runtime \
// RUN:   | FileCheck %s --check-prefixes=CHECK,SIGNED,UNSIGNED
//
// C retains signed-overflow divrem and arithmetic checks.
// RUN: %clang_cc1 %s -x c -triple x86_64-unknown-linux-gnu -emit-llvm -o - \
// RUN:   -disable-llvm-passes -DKERNEL= -DGLOBAL= \
// RUN:   -fsanitize=signed-integer-overflow -fsanitize-minimal-runtime \
// RUN:   | FileCheck %s --check-prefixes=CHECK,ARITH,SIGNED

// Signed: both the divide-by-zero and the INT_MIN / -1 arm would apply.
// CHECK-LABEL: divrem_signed
// SIGNED: call{{.*}} @__ubsan_handle_divrem_overflow
// CHECK: sdiv i32
// SIGNED: call{{.*}} @__ubsan_handle_divrem_overflow
// CHECK: srem i32
KERNEL void divrem_signed(GLOBAL int *a, GLOBAL int *b, GLOBAL int *out) {
  out[0] = a[0] / b[0];
  out[1] = a[0] % b[0];
}

// Unsigned: only the divide-by-zero arm would apply.
// CHECK-LABEL: divrem_unsigned
// UNSIGNED: call{{.*}} @__ubsan_handle_divrem_overflow
// CHECK: udiv i32
// UNSIGNED: call{{.*}} @__ubsan_handle_divrem_overflow
// CHECK: urem i32
KERNEL void divrem_unsigned(GLOBAL unsigned int *a, GLOBAL unsigned int *b,
                            GLOBAL unsigned int *out) {
  out[0] = a[0] / b[0];
  out[1] = a[0] % b[0];
}

// Compound assignment reaches the same emission path. This is the form the
// OpenCL-CTS integer_divideAssign and integer_moduloAssign kernels use.
// CHECK-LABEL: compound_divrem
// SIGNED: call{{.*}} @__ubsan_handle_divrem_overflow
// CHECK: sdiv i32
// SIGNED: call{{.*}} @__ubsan_handle_divrem_overflow
// CHECK: srem i32
KERNEL void compound_divrem(GLOBAL int *a, GLOBAL int *out) {
  out[0] /= a[0];
  out[1] %= a[0];
}

// Integer vectors are not instrumented today: the check requires a scalar
// integer type. The suppressed runs still must not find a handler here, so a
// later change that instruments vectors outside the guarded path fails this
// test. The OpenCL-CTS integer_ops kernels run every vector width.
// CHECK-LABEL: vector_divrem
// CHECK: sdiv <4 x i32>
// CHECK: srem <4 x i32>
typedef int vint4 __attribute__((ext_vector_type(4)));
KERNEL void vector_divrem(GLOBAL vint4 *a, GLOBAL vint4 *b, GLOBAL vint4 *out) {
  out[0] = a[0] / b[0];
  out[1] = a[0] % b[0];
}

// Controls: signed overflow in +, - and * must still be checked.
// CHECK-LABEL: add_still_checked
// ARITH: call{{.*}} @__ubsan_handle_add_overflow
KERNEL void add_still_checked(GLOBAL int *a, GLOBAL int *b, GLOBAL int *out) {
  out[0] = a[0] + b[0];
}

// CHECK-LABEL: sub_still_checked
// ARITH: call{{.*}} @__ubsan_handle_sub_overflow
KERNEL void sub_still_checked(GLOBAL int *a, GLOBAL int *b, GLOBAL int *out) {
  out[0] = a[0] - b[0];
}

// CHECK-LABEL: mul_still_checked
// ARITH: call{{.*}} @__ubsan_handle_mul_overflow
KERNEL void mul_still_checked(GLOBAL int *a, GLOBAL int *b, GLOBAL int *out) {
  out[0] = a[0] * b[0];
}
