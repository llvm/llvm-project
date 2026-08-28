// Test that -ffp-contract=on fuses a*b+c / a*b-c into cir.fmuladd and that
// -ffp-contract=off does not. The CIR-lowered and classic CodeGen LLVM IR
// match here, so both feed the LLVM-* prefixes.

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -ffp-contract=on -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR-ON
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -ffp-contract=off -emit-cir %s -o %t-off.cir
// RUN: FileCheck --input-file=%t-off.cir %s -check-prefix=CIR-OFF

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -ffp-contract=on -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM-ON
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -ffp-contract=off -emit-llvm %s -o %t-off.ll
// RUN: FileCheck --input-file=%t-off.ll %s -check-prefix=LLVM-OFF

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -ffp-contract=on -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM-ON
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -ffp-contract=off -emit-llvm %s -o %t-off.ll
// RUN: FileCheck --input-file=%t-off.ll %s -check-prefix=LLVM-OFF

// Under strict FP the fused op carries an fenv attribute and lowers to the
// constrained fmuladd intrinsic.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -ffp-contract=on -fexperimental-strict-floating-point -ffp-exception-behavior=strict -emit-cir %s -o %t-strict.cir
// RUN: FileCheck --input-file=%t-strict.cir %s -check-prefix=CIR-STRICT
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -ffp-contract=on -fexperimental-strict-floating-point -ffp-exception-behavior=strict -emit-llvm %s -o %t-strict.ll
// RUN: FileCheck --input-file=%t-strict.ll %s -check-prefix=LLVM-STRICT
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -ffp-contract=on -fexperimental-strict-floating-point -ffp-exception-behavior=strict -emit-llvm %s -o %t-strict-ogcg.ll
// RUN: FileCheck --input-file=%t-strict-ogcg.ll %s -check-prefix=LLVM-STRICT

// a * b + c  =>  fmuladd(a, b, c)
float fmuladd_add(float a, float b, float c) {
  return a * b + c;
}
// CIR-ON-LABEL: cir.func {{.*}}@fmuladd_add
// CIR-ON: cir.fmuladd %{{.*}}, %{{.*}}, %{{.*}} : !cir.float
// CIR-ON-NOT: cir.fmul

// CIR-OFF-LABEL: cir.func {{.*}}@fmuladd_add
// CIR-OFF: cir.fmul %{{.*}}, %{{.*}} : !cir.float
// CIR-OFF: cir.fadd %{{.*}}, %{{.*}} : !cir.float
// CIR-OFF-NOT: cir.fmuladd

// LLVM-ON-LABEL: @fmuladd_add
// LLVM-ON: call float @llvm.fmuladd.f32
// LLVM-OFF-LABEL: @fmuladd_add
// LLVM-OFF: fmul float
// LLVM-OFF: fadd float


// c + a * b  =>  fmuladd(a, b, c)  (mul on the RHS)
float fmuladd_add_rhs(float a, float b, float c) {
  return c + a * b;
}
// CIR-ON-LABEL: cir.func {{.*}}@fmuladd_add_rhs
// CIR-ON: cir.fmuladd %{{.*}}, %{{.*}}, %{{.*}} : !cir.float

// LLVM-ON-LABEL: @fmuladd_add_rhs
// LLVM-ON: call float @llvm.fmuladd.f32

// a * b - c  =>  fmuladd(a, b, -c)
float fmuladd_sub(float a, float b, float c) {
  return a * b - c;
}
// CIR-ON-LABEL: cir.func {{.*}}@fmuladd_sub
// CIR-ON: %[[NEG:.*]] = cir.fneg %{{.*}} : !cir.float
// CIR-ON: cir.fmuladd %{{.*}}, %{{.*}}, %[[NEG]] : !cir.float

// LLVM-ON-LABEL: @fmuladd_sub
// LLVM-ON: %[[NEG:.*]] = fneg float
// LLVM-ON: call float @llvm.fmuladd.f32(float %{{.*}}, float %{{.*}}, float %[[NEG]])

// If the mul result is used elsewhere, it must NOT be fused.
float no_fmuladd_reused_mul(float a, float b, float c, float *p) {
  float m = a * b;
  *p = m;
  return m + c;
}
// CIR-ON-LABEL: cir.func {{.*}}@no_fmuladd_reused_mul
// CIR-ON: cir.fmul %{{.*}}, %{{.*}} : !cir.float
// CIR-ON: cir.fadd %{{.*}}, %{{.*}} : !cir.float
// CIR-ON-NOT: cir.fmuladd

// LLVM-ON-LABEL: @no_fmuladd_reused_mul
// LLVM-ON: fmul float
// LLVM-ON: fadd float
// LLVM-ON-NOT: call float @llvm.fmuladd.f32

// Vector: a * b + c  =>  fmuladd on the vector type.
typedef float float4 __attribute__((ext_vector_type(4)));
float4 fmuladd_vec(float4 a, float4 b, float4 c) {
  return a * b + c;
}
// CIR-ON-LABEL: cir.func {{.*}}@fmuladd_vec
// CIR-ON: cir.fmuladd %{{.*}}, %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float>

// LLVM-ON-LABEL: @fmuladd_vec
// LLVM-ON: call <4 x float> @llvm.fmuladd.v4f32

// Strict FP: fused op carries an fenv attr, lowering to the constrained
// fmuladd intrinsic.
float fmuladd_strict(float a, float b, float c) {
  return a * b + c;
}
// CIR-STRICT-LABEL: cir.func {{.*}}@fmuladd_strict
// CIR-STRICT: cir.fmuladd %{{.*}}, %{{.*}}, %{{.*}} : !cir.float {fenv = #cir.fenv<{{.*}}strict_except = true>}
// LLVM-STRICT-LABEL: @fmuladd_strict
// LLVM-STRICT: call float @llvm.experimental.constrained.fmuladd.f32

// Strict FP with a negated addend: the fmuladd carries the mul's fenv while
// the fneg (which takes none) lowers to a plain fneg.
float fmuladd_sub_strict(float a, float b, float c) {
  return a * b - c;
}
// CIR-STRICT-LABEL: cir.func {{.*}}@fmuladd_sub_strict
// CIR-STRICT: cir.fneg %{{.*}} : !cir.float
// CIR-STRICT: cir.fmuladd %{{.*}}, %{{.*}}, %{{.*}} : !cir.float {fenv = #cir.fenv<{{.*}}strict_except = true>}
// LLVM-STRICT-LABEL: @fmuladd_sub_strict
// LLVM-STRICT: fneg float
// LLVM-STRICT: call float @llvm.experimental.constrained.fmuladd.f32

// Compound assignment routes through emitAdd/emitSub, so += and -= fuse too.
float fmuladd_add_assign(float x, float a, float b) {
  x += a * b;
  return x;
}
// CIR-ON-LABEL: cir.func {{.*}}@fmuladd_add_assign
// CIR-ON: cir.fmuladd %{{.*}}, %{{.*}}, %{{.*}} : !cir.float
// LLVM-ON-LABEL: @fmuladd_add_assign
// LLVM-ON: call float @llvm.fmuladd.f32

// x -= a * b picks negMul off isSub with the mul on the RHS.
float fmuladd_sub_assign(float x, float a, float b) {
  x -= a * b;
  return x;
}
// CIR-ON-LABEL: cir.func {{.*}}@fmuladd_sub_assign
// CIR-ON: cir.fneg %{{.*}} : !cir.float
// CIR-ON: cir.fmuladd %{{.*}}, %{{.*}}, %{{.*}} : !cir.float
// LLVM-ON-LABEL: @fmuladd_sub_assign
// LLVM-ON: fneg float
// LLVM-ON: call float @llvm.fmuladd.f32
