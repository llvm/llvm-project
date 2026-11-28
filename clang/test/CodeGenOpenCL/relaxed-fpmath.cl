// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s -check-prefix=NORMAL
// RUN: %clang_cc1 %s -emit-llvm -cl-fast-relaxed-math -menable-no-nans -menable-no-infs -mreassociate -freciprocal-math -fapprox-func -fno-signed-zeros -o - | FileCheck %s -check-prefix=FAST
// RUN: %clang_cc1 %s -emit-llvm -menable-no-infs -menable-no-nans -cl-finite-math-only -o - | FileCheck %s -check-prefix=FINITE
// RUN: %clang_cc1 %s -emit-llvm -cl-unsafe-math-optimizations -o - | FileCheck %s -check-prefix=UNSAFE
// RUN: %clang_cc1 %s -emit-llvm -cl-mad-enable -o - | FileCheck %s -check-prefixes=MAD,PRAGMA
// RUN: %clang_cc1 %s -emit-llvm -cl-mad-enable -ffp-contract=off -o - | FileCheck %s -check-prefix=STRICT
// RUN: %clang_cc1 %s -emit-llvm -cl-no-signed-zeros -o - | FileCheck %s -check-prefix=NOSIGNED

// Check the fp options are correct with PCH.
// RUN: %clang_cc1 %s -DGEN_PCH=1 -finclude-default-header -triple spir-unknown-unknown -emit-pch -o %t.pch
// RUN: %clang_cc1 %s -include-pch %t.pch -fno-validate-pch -emit-llvm -o - | FileCheck %s -check-prefix=NORMAL
// RUN: %clang_cc1 %s -include-pch %t.pch -fno-validate-pch -emit-llvm -cl-fast-relaxed-math -menable-no-nans -menable-no-infs -mreassociate -freciprocal-math -fapprox-func -fno-signed-zeros -o - | FileCheck %s -check-prefix=FAST
// RUN: %clang_cc1 %s -include-pch %t.pch -fno-validate-pch -emit-llvm -menable-no-infs -menable-no-nans -cl-finite-math-only -o - | FileCheck %s -check-prefix=FINITE
// RUN: %clang_cc1 %s -include-pch %t.pch -fno-validate-pch -emit-llvm -cl-unsafe-math-optimizations -o - | FileCheck %s -check-prefix=UNSAFE
// RUN: %clang_cc1 %s -include-pch %t.pch -fno-validate-pch -emit-llvm -cl-no-signed-zeros -o - | FileCheck %s -check-prefix=NOSIGNED

#if !GEN_PCH

float spscalardiv(float a, float b) {
  // CHECK: @spscalardiv(

  // NORMAL: fdiv float
  // FAST: fdiv fast float
  // FINITE: fdiv nnan ninf float
  // UNSAFE: fdiv reassoc nsz arcp contract afn float
  // NOSIGNED: fdiv nsz float
  return a / b;
}

float mad_enable_separate_ops(float a, float b, float c, float d) {
  // NORMAL-LABEL: @mad_enable_separate_ops(
  // NORMAL: fmul float
  // NORMAL: fadd float
  // NORMAL: fsub float
  // MAD-LABEL: @mad_enable_separate_ops(
  // MAD: fmul contract float
  // MAD: fadd contract float
  // MAD: fsub contract float
  // STRICT-LABEL: @mad_enable_separate_ops(
  // STRICT: fmul float
  // STRICT: fadd float
  // STRICT: fsub float
  float product = a * b;
  float sum = product + c;
  return sum - d;
}

float mad_enable_same_expression(float a, float b, float c) {
  // NORMAL-LABEL: @mad_enable_same_expression(
  // NORMAL: call float @llvm.fmuladd.f32
  // MAD-LABEL: @mad_enable_same_expression(
  // MAD: call float @llvm.fmuladd.f32
  // STRICT-LABEL: @mad_enable_same_expression(
  // STRICT: fmul float
  // STRICT: fadd float
  return a * b + c;
}

#pragma OPENCL FP_CONTRACT OFF
float mad_enable_pragma_off(float a, float b, float c) {
  // PRAGMA-LABEL: @mad_enable_pragma_off(
  // PRAGMA: fmul float
  // PRAGMA: fadd float
  return a * b + c;
}
#pragma OPENCL FP_CONTRACT ON

#pragma OPENCL FP_CONTRACT OFF
float mad_enable_pragma_off_separate_ops(float a, float b, float c, float d) {
  // PRAGMA-LABEL: @mad_enable_pragma_off_separate_ops(
  // PRAGMA: fmul float
  // PRAGMA: fadd float
  // PRAGMA: fsub float
  float product = a * b;
  float sum = product + c;
  return sum - d;
}
#pragma OPENCL FP_CONTRACT ON
// CHECK: attributes

// NORMAL-NOT: "no-signed-zeros-fp-math"

// FAST: "no-signed-zeros-fp-math"="true"

// FINITE-NOT: "no-signed-zeros-fp-math"

// UNSAFE: "no-signed-zeros-fp-math"="true"

// NOSIGNED: "no-signed-zeros-fp-math"="true"

#else
// Undefine this to avoid putting it in the PCH.
#undef GEN_PCH
#endif
