// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fexperimental-strict-floating-point -ffp-exception-behavior=strict -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fexperimental-strict-floating-point -ffp-exception-behavior=strict -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fexperimental-strict-floating-point -ffp-exception-behavior=strict -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

// ClangIR and classic codegen share the LLVM lines. They end before the
// call-site attributes.
// FIXME: ClangIR does not mark the calls in a strictfp function `strictfp`.

typedef bool v4b __attribute__((ext_vector_type(4)));
typedef float v4f __attribute__((ext_vector_type(4)));

// The command line makes exceptions strict and keeps rounding to nearest.
float bool_to_float(bool b) { return b; }

// CIR-LABEL: cir.func {{.*}}@_Z13bool_to_floatb
// CIR: cir.cast bool_to_float %{{.*}} : !cir.bool -> !cir.float fenv<{{.*}}except_mode = unknown, strict_except = true>

// LLVM-LABEL: define {{.*}}float @_Z13bool_to_floatb
// LLVM: call float @llvm.experimental.constrained.uitofp.f32.i1(i1 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")

double bool_to_double_fenv_access(bool b) {
#pragma STDC FENV_ACCESS ON
  return b;
}

// CIR-LABEL: cir.func {{.*}}@_Z26bool_to_double_fenv_accessb
// CIR: cir.cast bool_to_float %{{.*}} : !cir.bool -> !cir.double fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = true>

// LLVM-LABEL: define {{.*}}double @_Z26bool_to_double_fenv_accessb
// LLVM: call double @llvm.experimental.constrained.uitofp.f64.i1(i1 %{{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")

void vec_bool_to_float() {
  v4b b;
  v4f f = __builtin_convertvector(b, v4f);
}

// CIR-LABEL: cir.func {{.*}}@_Z17vec_bool_to_floatv
// CIR: cir.cast bool_to_float %{{.*}} : !cir.vector<4 x !cir.bool> -> !cir.vector<4 x !cir.float> fenv<{{.*}}except_mode = unknown, strict_except = true>

// LLVM-LABEL: define {{.*}}void @_Z17vec_bool_to_floatv
// LLVM: call <4 x float> @llvm.experimental.constrained.uitofp.v4f32.v4i1(<4 x i1> %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
