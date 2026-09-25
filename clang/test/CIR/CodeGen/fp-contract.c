// Test that -ffp-contract=on fuses a*b+c / a*b-c into cir.fmuladd and that
// -ffp-contract=off does not. -ffp-contract=fast sets `contract` on the fmul and
// fadd instead. The CIR-lowered and classic CodeGen LLVM IR match here, so both
// feed the LLVM-* prefixes.

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

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -ffp-contract=fast -emit-cir %s -o %t-fast.cir
// RUN: FileCheck --input-file=%t-fast.cir %s -check-prefix=CIR-FAST
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -ffp-contract=fast -emit-llvm %s -o %t-fast.ll
// RUN: FileCheck --input-file=%t-fast.ll %s -check-prefix=LLVM-FAST
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -ffp-contract=fast -emit-llvm %s -o %t-fast-ogcg.ll
// RUN: FileCheck --input-file=%t-fast-ogcg.ll %s -check-prefix=LLVM-FAST

// -ffp-contract=fast-honor-pragmas matches -ffp-contract=fast here.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -ffp-contract=fast-honor-pragmas -emit-cir %s -o %t-fhp.cir
// RUN: FileCheck --input-file=%t-fhp.cir %s -check-prefix=CIR-FAST
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -ffp-contract=fast-honor-pragmas -emit-llvm %s -o %t-fhp.ll
// RUN: FileCheck --input-file=%t-fhp.ll %s -check-prefix=LLVM-FAST
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -ffp-contract=fast-honor-pragmas -emit-llvm %s -o %t-fhp-ogcg.ll
// RUN: FileCheck --input-file=%t-fhp-ogcg.ll %s -check-prefix=LLVM-FAST

// Under strict FP the fused op carries an fenv attribute and lowers to the
// constrained fmuladd intrinsic.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -ffp-contract=on -fexperimental-strict-floating-point -ffp-exception-behavior=strict -emit-cir %s -o %t-strict.cir
// RUN: FileCheck --input-file=%t-strict.cir %s -check-prefix=CIR-STRICT
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -ffp-contract=on -fexperimental-strict-floating-point -ffp-exception-behavior=strict -emit-llvm %s -o %t-strict.ll
// RUN: FileCheck --input-file=%t-strict.ll %s -check-prefix=LLVM-STRICT
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -ffp-contract=on -fexperimental-strict-floating-point -ffp-exception-behavior=strict -emit-llvm %s -o %t-strict-ogcg.ll
// RUN: FileCheck --input-file=%t-strict-ogcg.ll %s -check-prefix=LLVM-STRICT

// Under strict FP with -ffp-contract=fast, the constrained intrinsics carry
// `contract`.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -ffp-contract=fast -fexperimental-strict-floating-point -ffp-exception-behavior=strict -emit-cir %s -o %t-strict-fast.cir
// RUN: FileCheck --input-file=%t-strict-fast.cir %s -check-prefix=CIR-STRICT-FAST
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -ffp-contract=fast -fexperimental-strict-floating-point -ffp-exception-behavior=strict -emit-llvm %s -o %t-strict-fast.ll
// RUN: FileCheck --input-file=%t-strict-fast.ll %s -check-prefix=LLVM-STRICT-FAST
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -ffp-contract=fast -fexperimental-strict-floating-point -ffp-exception-behavior=strict -emit-llvm %s -o %t-strict-fast-ogcg.ll
// RUN: FileCheck --input-file=%t-strict-fast-ogcg.ll %s -check-prefix=LLVM-STRICT-FAST

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

// CIR-FAST-LABEL: cir.func {{.*}}@fmuladd_add
// CIR-FAST: cir.fmul %{{.*}}, %{{.*}} : !cir.float {fastmath_flags = #cir.fastmath<contract>}
// CIR-FAST: cir.fadd %{{.*}}, %{{.*}} : !cir.float {fastmath_flags = #cir.fastmath<contract>}
// CIR-FAST-NOT: cir.fmuladd

// LLVM-FAST-LABEL: @fmuladd_add
// LLVM-FAST: fmul contract float
// LLVM-FAST: fadd contract float


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

// CIR-FAST-LABEL: cir.func {{.*}}@fmuladd_vec
// CIR-FAST: cir.fmul %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> {fastmath_flags = #cir.fastmath<contract>}
// CIR-FAST: cir.fadd %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> {fastmath_flags = #cir.fastmath<contract>}

// LLVM-FAST-LABEL: @fmuladd_vec
// LLVM-FAST: fmul contract <4 x float>
// LLVM-FAST: fadd contract <4 x float>

// Strict FP: fused op carries an fenv attr, lowering to the constrained
// fmuladd intrinsic.
float fmuladd_strict(float a, float b, float c) {
  return a * b + c;
}
// CIR-STRICT-LABEL: cir.func {{.*}}@fmuladd_strict
// CIR-STRICT: cir.fmuladd %{{.*}}, %{{.*}}, %{{.*}} : !cir.float {fenv = #cir.fenv<{{.*}}strict_except = true>}
// LLVM-STRICT-LABEL: @fmuladd_strict
// LLVM-STRICT: call float @llvm.experimental.constrained.fmuladd.f32

// CIR-STRICT-FAST-LABEL: cir.func {{.*}}@fmuladd_strict
// CIR-STRICT-FAST: cir.fmul %{{.*}}, %{{.*}} : !cir.float
// CIR-STRICT-FAST-SAME: fastmath_flags = #cir.fastmath<contract>
// CIR-STRICT-FAST: cir.fadd %{{.*}}, %{{.*}} : !cir.float
// CIR-STRICT-FAST-SAME: fastmath_flags = #cir.fastmath<contract>

// LLVM-STRICT-FAST-LABEL: @fmuladd_strict
// LLVM-STRICT-FAST: call contract float @llvm.experimental.constrained.fmul.f32
// LLVM-STRICT-FAST: call contract float @llvm.experimental.constrained.fadd.f32

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

// The pragma turns contraction off for this function only.
float contract_pragma_off(float a, float b, float c) {
#pragma clang fp contract(off)
  return a * b + c;
}
// CIR-FAST-LABEL: cir.func {{.*}}@contract_pragma_off
// CIR-FAST-NOT: #cir.fastmath
// CIR-FAST: cir.return

// LLVM-FAST-LABEL: @contract_pragma_off
// LLVM-FAST: fmul float
// LLVM-FAST: fadd float

// -ffp-contract=fast also allows contraction across statements.
float contract_across_stmt(float a, float b, float c) {
  float t = a * b;
  return t + c;
}
// CIR-FAST-LABEL: cir.func {{.*}}@contract_across_stmt
// CIR-FAST: cir.fmul %{{.*}}, %{{.*}} : !cir.float {fastmath_flags = #cir.fastmath<contract>}
// CIR-FAST: cir.fadd %{{.*}}, %{{.*}} : !cir.float {fastmath_flags = #cir.fastmath<contract>}

// LLVM-FAST-LABEL: @contract_across_stmt
// LLVM-FAST: fmul contract float
// LLVM-FAST: fadd contract float

// Nested pragmas: each scope sets its own flags and the enclosing ones are
// restored on exit.
float nested_pragmas(float a, float b, float c) {
  float r;
  {
#pragma STDC FP_CONTRACT OFF
    r = a * b + c;
    {
#pragma clang fp contract(fast)
      r = r * a + c;
    }
    r = r * b + c;
  }
  {
#pragma float_control(precise, on)
    r = r * a + b;
  }
  return r * c + a;
}
// CIR-FAST-LABEL: cir.func {{.*}}@nested_pragmas
// CIR-FAST: cir.fmul %{{.*}}, %{{.*}} : !cir.float{{( loc.*)?$}}
// CIR-FAST: cir.fadd %{{.*}}, %{{.*}} : !cir.float{{( loc.*)?$}}
// CIR-FAST: cir.fmul %{{.*}}, %{{.*}} : !cir.float {fastmath_flags = #cir.fastmath<contract>}
// CIR-FAST: cir.fadd %{{.*}}, %{{.*}} : !cir.float {fastmath_flags = #cir.fastmath<contract>}
// CIR-FAST: cir.fmul %{{.*}}, %{{.*}} : !cir.float{{( loc.*)?$}}
// CIR-FAST: cir.fadd %{{.*}}, %{{.*}} : !cir.float{{( loc.*)?$}}
// CIR-FAST: cir.fmuladd %{{.*}}, %{{.*}}, %{{.*}} : !cir.float{{( loc.*)?$}}
// CIR-FAST: cir.fmul %{{.*}}, %{{.*}} : !cir.float {fastmath_flags = #cir.fastmath<contract>}
// CIR-FAST: cir.fadd %{{.*}}, %{{.*}} : !cir.float {fastmath_flags = #cir.fastmath<contract>}

// float_control(precise, on) enables contraction within the statement even
// under -ffp-contract=off.
// CIR-OFF-LABEL: cir.func {{.*}}@nested_pragmas
// CIR-OFF: cir.fmuladd %{{.*}}, %{{.*}}, %{{.*}} : !cir.float

// LLVM-FAST-LABEL: @nested_pragmas
// LLVM-FAST: fmul float
// LLVM-FAST: fadd float
// LLVM-FAST: fmul contract float
// LLVM-FAST: fadd contract float
// LLVM-FAST: fmul float
// LLVM-FAST: fadd float
// LLVM-FAST: call float @llvm.fmuladd.f32
// LLVM-FAST: fmul contract float
// LLVM-FAST: fadd contract float
