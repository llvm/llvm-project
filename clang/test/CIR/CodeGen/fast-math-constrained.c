// Fast-math flags also apply to constrained arithmetic. Set the flags
// explicitly so the strict exception behavior is retained.

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fexperimental-strict-floating-point -ffp-exception-behavior=strict -ffp-contract=off -emit-cir %s -o - | FileCheck %s --check-prefixes=CIR,CIR-NONE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fexperimental-strict-floating-point -ffp-exception-behavior=strict -ffp-contract=off -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVM-NONE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fexperimental-strict-floating-point -ffp-exception-behavior=strict -ffp-contract=off -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVM-NONE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fexperimental-strict-floating-point -ffp-exception-behavior=strict -ffp-contract=off -menable-no-nans -menable-no-infs -emit-cir %s -o - | FileCheck %s --check-prefixes=CIR,CIR-FINITE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fexperimental-strict-floating-point -ffp-exception-behavior=strict -ffp-contract=off -menable-no-nans -menable-no-infs -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVM-FINITE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fexperimental-strict-floating-point -ffp-exception-behavior=strict -ffp-contract=off -menable-no-nans -menable-no-infs -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVM-FINITE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fexperimental-strict-floating-point -ffp-exception-behavior=strict -menable-no-nans -menable-no-infs -fno-signed-zeros -freciprocal-math -fapprox-func -mreassociate -ffp-contract=fast -emit-cir %s -o - | FileCheck %s --check-prefixes=CIR,CIR-FAST
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fexperimental-strict-floating-point -ffp-exception-behavior=strict -menable-no-nans -menable-no-infs -fno-signed-zeros -freciprocal-math -fapprox-func -mreassociate -ffp-contract=fast -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVM-FAST
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fexperimental-strict-floating-point -ffp-exception-behavior=strict -menable-no-nans -menable-no-infs -fno-signed-zeros -freciprocal-math -fapprox-func -mreassociate -ffp-contract=fast -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVM-FAST

float scalar(float a, float b, float c) {
  return ((a + b) * (a - b)) / c;
}
// CIR-LABEL: cir.func {{.*}}@scalar(
// LLVM-LABEL: define {{.*}}@scalar(
// CIR-NONE: cir.fadd %{{.*}}, %{{.*}} : !cir.float fenv<{{.*}}strict_except = true>
// CIR-NONE: cir.fsub %{{.*}}, %{{.*}} : !cir.float fenv<{{.*}}strict_except = true>
// CIR-NONE: cir.fmul %{{.*}}, %{{.*}} : !cir.float fenv<{{.*}}strict_except = true>
// CIR-NONE: cir.fdiv %{{.*}}, %{{.*}} : !cir.float fenv<{{.*}}strict_except = true>
// LLVM-NONE: call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-NONE: call float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-NONE: call float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-NONE: call float @llvm.experimental.constrained.fdiv.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CIR-FINITE: cir.fadd %{{.*}}, %{{.*}} : !cir.float fenv<{{.*}}strict_except = true> <fastmath_flags = [nnan, ninf]>
// CIR-FINITE: cir.fsub %{{.*}}, %{{.*}} : !cir.float fenv<{{.*}}strict_except = true> <fastmath_flags = [nnan, ninf]>
// CIR-FINITE: cir.fmul %{{.*}}, %{{.*}} : !cir.float fenv<{{.*}}strict_except = true> <fastmath_flags = [nnan, ninf]>
// CIR-FINITE: cir.fdiv %{{.*}}, %{{.*}} : !cir.float fenv<{{.*}}strict_except = true> <fastmath_flags = [nnan, ninf]>
// LLVM-FINITE: call nnan ninf float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-FINITE: call nnan ninf float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-FINITE: call nnan ninf float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-FINITE: call nnan ninf float @llvm.experimental.constrained.fdiv.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CIR-FAST: cir.fadd %{{.*}}, %{{.*}} : !cir.float fenv<{{.*}}strict_except = true> <fastmath_flags = [fast]>
// CIR-FAST: cir.fsub %{{.*}}, %{{.*}} : !cir.float fenv<{{.*}}strict_except = true> <fastmath_flags = [fast]>
// CIR-FAST: cir.fmul %{{.*}}, %{{.*}} : !cir.float fenv<{{.*}}strict_except = true> <fastmath_flags = [fast]>
// CIR-FAST: cir.fdiv %{{.*}}, %{{.*}} : !cir.float fenv<{{.*}}strict_except = true> <fastmath_flags = [fast]>
// LLVM-FAST: call fast float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-FAST: call fast float @llvm.experimental.constrained.fsub.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-FAST: call fast float @llvm.experimental.constrained.fmul.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-FAST: call fast float @llvm.experimental.constrained.fdiv.f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")

typedef float float4 __attribute__((ext_vector_type(4)));

float4 vector(float4 a, float4 b, float4 c) {
  return ((a + b) * (a - b)) / c;
}
// CIR-LABEL: cir.func {{.*}}@vector(
// LLVM-LABEL: define {{.*}}@vector(
// CIR-NONE: cir.fadd %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> fenv<{{.*}}strict_except = true>
// CIR-NONE: cir.fsub %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> fenv<{{.*}}strict_except = true>
// CIR-NONE: cir.fmul %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> fenv<{{.*}}strict_except = true>
// CIR-NONE: cir.fdiv %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> fenv<{{.*}}strict_except = true>
// LLVM-NONE: call <4 x float> @llvm.experimental.constrained.fadd.v4f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-NONE: call <4 x float> @llvm.experimental.constrained.fsub.v4f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-NONE: call <4 x float> @llvm.experimental.constrained.fmul.v4f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-NONE: call <4 x float> @llvm.experimental.constrained.fdiv.v4f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CIR-FINITE: cir.fadd %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> fenv<{{.*}}strict_except = true> <fastmath_flags = [nnan, ninf]>
// CIR-FINITE: cir.fsub %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> fenv<{{.*}}strict_except = true> <fastmath_flags = [nnan, ninf]>
// CIR-FINITE: cir.fmul %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> fenv<{{.*}}strict_except = true> <fastmath_flags = [nnan, ninf]>
// CIR-FINITE: cir.fdiv %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> fenv<{{.*}}strict_except = true> <fastmath_flags = [nnan, ninf]>
// LLVM-FINITE: call nnan ninf <4 x float> @llvm.experimental.constrained.fadd.v4f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-FINITE: call nnan ninf <4 x float> @llvm.experimental.constrained.fsub.v4f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-FINITE: call nnan ninf <4 x float> @llvm.experimental.constrained.fmul.v4f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-FINITE: call nnan ninf <4 x float> @llvm.experimental.constrained.fdiv.v4f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CIR-FAST: cir.fadd %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> fenv<{{.*}}strict_except = true> <fastmath_flags = [fast]>
// CIR-FAST: cir.fsub %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> fenv<{{.*}}strict_except = true> <fastmath_flags = [fast]>
// CIR-FAST: cir.fmul %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> fenv<{{.*}}strict_except = true> <fastmath_flags = [fast]>
// CIR-FAST: cir.fdiv %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float> fenv<{{.*}}strict_except = true> <fastmath_flags = [fast]>
// LLVM-FAST: call fast <4 x float> @llvm.experimental.constrained.fadd.v4f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-FAST: call fast <4 x float> @llvm.experimental.constrained.fsub.v4f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-FAST: call fast <4 x float> @llvm.experimental.constrained.fmul.v4f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM-FAST: call fast <4 x float> @llvm.experimental.constrained.fdiv.v4f32({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
