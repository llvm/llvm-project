// RUN: %clang_cc1 -triple x86_64-unknown-unknown %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown %s -ffp-exception-behavior=maytrap -emit-llvm -o - | FileCheck %s

// Test codegen of constrained floating point to bool conversion
//
// Test that the constrained intrinsics are picking up the exception
// metadata from the AST instead of the global default from the command line.

#pragma float_control(except, on)

void eMaisUma() {
  double t[1];
  if (*t)
    return;
// CHECK: call i1 @llvm.experimental.constrained.fcmp.f64(double %{{.*}}, double 0.000000e+00, metadata !"une", metadata !"fpexcept.strict")
}

