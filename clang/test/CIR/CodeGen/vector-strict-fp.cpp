// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -ffp-contract=on -fexperimental-strict-floating-point -ffp-exception-behavior=strict -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -ffp-contract=on -fexperimental-strict-floating-point -ffp-exception-behavior=strict -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefixes=LLVM,SHARED \
// RUN:   --implicit-check-not="{{= (sitofp|uitofp|fptosi|fptoui|fptrunc|fpext|fcmp) }}"
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -ffp-contract=on -fexperimental-strict-floating-point -ffp-exception-behavior=strict -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefixes=OGCG,SHARED

typedef float v4f __attribute__((ext_vector_type(4)));
typedef int v4i __attribute__((ext_vector_type(4)));
typedef unsigned v4u __attribute__((ext_vector_type(4)));
typedef double v4d __attribute__((ext_vector_type(4)));
typedef bool v4b __attribute__((ext_vector_type(4)));

void vec_logical_not() {
  v4f a;
  v4i b = !a;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!cir.vector<4 x !cir.float>>
// CIR: %[[B_ADDR:.*]] = cir.alloca "b" {{.*}} init : !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float>
// CIR: %[[CONST_ZERO:.*]] = cir.const #cir.zero : !cir.vector<4 x !cir.float>
// CIR: %[[RESULT:.*]] = cir.vec.fcmp(oeq, %[[TMP_A]], %[[CONST_ZERO]]) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !s32i> fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>
// CIR: cir.store {{.*}} %[[RESULT]], %[[B_ADDR]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>

// SHARED: %[[A_ADDR:.*]] = alloca <4 x float>, align 16
// SHARED: %[[B_ADDR:.*]] = alloca <4 x i32>, align 16
// SHARED: %[[TMP_A:.*]] = load <4 x float>, ptr %[[A_ADDR]], align 16
// SHARED: %[[RESULT:.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %[[TMP_A]], <4 x float> zeroinitializer, metadata !"oeq", metadata !"fpexcept.strict")
// SHARED: %[[RESULT_V4I:.*]] = sext <4 x i1> %[[RESULT]] to <4 x i32>
// SHARED: store <4 x i32> %[[RESULT_V4I]], ptr %[[B_ADDR]], align 16

void vec_logical_or() {
  v4f a;
  v4f b;
  v4i r = a || b;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!cir.vector<4 x !cir.float>>
// CIR: %[[B_ADDR:.*]] = cir.alloca "b" {{.*}} : !cir.ptr<!cir.vector<4 x !cir.float>>
// CIR: %[[R_ADDR:.*]] = cir.alloca "r" {{.*}} init : !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[CONST_ZERO:.*]] = cir.const #cir.zero : !cir.vector<4 x !cir.float>
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float>
// CIR: %[[TMP_B:.*]] = cir.load {{.*}} %[[B_ADDR]] : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float>
// CIR: %[[A_NE_ZERO:.*]] = cir.vec.fcmp(une, %[[TMP_A]], %[[CONST_ZERO]]) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !s32i> fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>
// CIR: %[[B_NE_ZERO:.*]] = cir.vec.fcmp(une, %[[TMP_B]], %[[CONST_ZERO]]) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !s32i> fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>
// CIR: %[[RESULT:.*]] = cir.or %[[A_NE_ZERO]], %[[B_NE_ZERO]] : !cir.vector<4 x !s32i>
// CIR: cir.store {{.*}} %[[RESULT]], %[[R_ADDR]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>

// SHARED: %[[A_ADDR:.*]] = alloca <4 x float>, align 16
// SHARED: %[[B_ADDR:.*]] = alloca <4 x float>, align 16
// SHARED: %[[R_ADDR:.*]] = alloca <4 x i32>, align 16
// SHARED: %[[TMP_A:.*]] = load <4 x float>, ptr %[[A_ADDR]], align 16
// SHARED: %[[TMP_B:.*]] = load <4 x float>, ptr %[[B_ADDR]], align 16

// In CIR IR vec compare will perform sext cast -> logical or

// SHARED: %[[A_NE_ZERO:.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %[[TMP_A]], <4 x float> zeroinitializer, metadata !"une", metadata !"fpexcept.strict")
// LLVM: %[[A_NE_ZERO_VEC:.*]] = sext <4 x i1> %[[A_NE_ZERO]] to <4 x i32>
// SHARED: %[[B_NE_ZERO:.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %[[TMP_B]], <4 x float> zeroinitializer, metadata !"une", metadata !"fpexcept.strict")
// LLVM: %[[B_NE_ZERO_VEC:.*]] = sext <4 x i1> %[[B_NE_ZERO]] to <4 x i32>
// LLVM: %[[RESULT_VEC:.*]] = or <4 x i32> %[[A_NE_ZERO_VEC]], %[[B_NE_ZERO_VEC]]

// In OG it's perform compare -> logical or -> sext

// OGCG: %[[RESULT:.*]] = or <4 x i1> %[[A_NE_ZERO]], %[[B_NE_ZERO]]
// OGCG: %[[RESULT_VEC:.*]] = sext <4 x i1> %2 to <4 x i32>

// SHARED: store <4 x i32> %[[RESULT_VEC]], ptr %[[R_ADDR]], align 16

void vec_logical_and() {
  v4f a;
  v4f b;
  v4i r = a && b;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!cir.vector<4 x !cir.float>>
// CIR: %[[B_ADDR:.*]] = cir.alloca "b" {{.*}} : !cir.ptr<!cir.vector<4 x !cir.float>>
// CIR: %[[R_ADDR:.*]] = cir.alloca "r" {{.*}} init : !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[CONST_ZERO:.*]] = cir.const #cir.zero : !cir.vector<4 x !cir.float>
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float>
// CIR: %[[TMP_B:.*]] = cir.load {{.*}} %[[B_ADDR]] : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float>
// CIR: %[[A_NE_ZERO:.*]] = cir.vec.fcmp(une, %[[TMP_A]], %[[CONST_ZERO]]) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !s32i> fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>
// CIR: %[[B_NE_ZERO:.*]] = cir.vec.fcmp(une, %[[TMP_B]], %[[CONST_ZERO]]) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !s32i> fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>
// CIR: %[[RESULT:.*]] = cir.and %[[A_NE_ZERO]], %[[B_NE_ZERO]] : !cir.vector<4 x !s32i>
// CIR: cir.store {{.*}} %[[RESULT]], %[[R_ADDR]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>

// SHARED: %[[A_ADDR:.*]] = alloca <4 x float>, align 16
// SHARED: %[[B_ADDR:.*]] = alloca <4 x float>, align 16
// SHARED: %[[R_ADDR:.*]] = alloca <4 x i32>, align 16
// SHARED: %[[TMP_A:.*]] = load <4 x float>, ptr %[[A_ADDR]], align 16
// SHARED: %[[TMP_B:.*]] = load <4 x float>, ptr %[[B_ADDR]], align 16

// In CIR IR vec compare will perform sext cast -> logical and

// FIXME: ClangIR does not put the strictfp attribute on these calls.
// SHARED: %[[A_NE_ZERO:.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %[[TMP_A]], <4 x float> zeroinitializer, metadata !"une", metadata !"fpexcept.strict")
// OGCG-SAME: #2
// LLVM: %[[A_NE_ZERO_VEC:.*]] = sext <4 x i1> %[[A_NE_ZERO]] to <4 x i32>
// SHARED: %[[B_NE_ZERO:.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %[[TMP_B]], <4 x float> zeroinitializer, metadata !"une", metadata !"fpexcept.strict")
// OGCG-SAME: #2
// LLVM: %[[B_NE_ZERO_VEC:.*]] = sext <4 x i1> %[[B_NE_ZERO]] to <4 x i32>
// LLVM: %[[RESULT_VEC:.*]] = and <4 x i32> %[[A_NE_ZERO_VEC]], %[[B_NE_ZERO_VEC]]

// In OG it's perform compare -> logical and -> sext

// OGCG: %[[RESULT:.*]] = and <4 x i1> %[[A_NE_ZERO]], %[[B_NE_ZERO]]
// OGCG: %[[RESULT_VEC:.*]] = sext <4 x i1> %2 to <4 x i32>

// SHARED: store <4 x i32> %[[RESULT_VEC]], ptr %[[R_ADDR]], align 16

// __builtin_convertvector converts in the floating-point environment that the
// pragmas set, not in the one from the command line.
void vec_convert() {
#pragma clang fp exceptions(maytrap)
#pragma STDC FENV_ACCESS ON
  v4i i;
  v4u u;
  v4f f;
  v4d d;
  v4b b;
  v4f int_to_float = __builtin_convertvector(i, v4f);
  v4f uint_to_float = __builtin_convertvector(u, v4f);
  v4i float_to_int = __builtin_convertvector(f, v4i);
  v4u float_to_uint = __builtin_convertvector(f, v4u);
  v4f float_trunc = __builtin_convertvector(d, v4f);
  v4d float_ext = __builtin_convertvector(f, v4d);
  v4b float_to_bool = __builtin_convertvector(f, v4b);
  v4f bool_to_float = __builtin_convertvector(b, v4f);
}

// CIR-LABEL: cir.func {{.*}}@_Z11vec_convertv(
// CIR: cir.cast int_to_float %{{.*}} : !cir.vector<4 x !s32i> -> !cir.vector<4 x !cir.float> fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = false>
// CIR: cir.cast int_to_float %{{.*}} : !cir.vector<4 x !u32i> -> !cir.vector<4 x !cir.float> fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = false>
// CIR: cir.cast float_to_int %{{.*}} : !cir.vector<4 x !cir.float> -> !cir.vector<4 x !s32i> fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = false>
// CIR: cir.cast float_to_int %{{.*}} : !cir.vector<4 x !cir.float> -> !cir.vector<4 x !u32i> fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = false>
// CIR: cir.cast floating %{{.*}} : !cir.vector<4 x !cir.double> -> !cir.vector<4 x !cir.float> fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = false>
// CIR: cir.cast floating %{{.*}} : !cir.vector<4 x !cir.float> -> !cir.vector<4 x !cir.double> fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = false>
// CIR: cir.cast float_to_bool %{{.*}} : !cir.vector<4 x !cir.float> -> !cir.vector<4 x !cir.bool> fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = false>
// CIR: cir.cast bool_to_float %{{.*}} : !cir.vector<4 x !cir.bool> -> !cir.vector<4 x !cir.float> fenv<dynamic_rounding_mode = unknown, except_mode = unknown, strict_except = false>

// SHARED-LABEL: define {{.*}}@_Z11vec_convertv(
// SHARED: call <4 x float> @llvm.experimental.constrained.sitofp.v4f32.v4i32(<4 x i32> %{{.*}}, metadata !"round.dynamic", metadata !"fpexcept.maytrap")
// SHARED: call <4 x float> @llvm.experimental.constrained.uitofp.v4f32.v4i32(<4 x i32> %{{.*}}, metadata !"round.dynamic", metadata !"fpexcept.maytrap")
// SHARED: call <4 x i32> @llvm.experimental.constrained.fptosi.v4i32.v4f32(<4 x float> %{{.*}}, metadata !"fpexcept.maytrap")
// SHARED: call <4 x i32> @llvm.experimental.constrained.fptoui.v4i32.v4f32(<4 x float> %{{.*}}, metadata !"fpexcept.maytrap")
// SHARED: call <4 x float> @llvm.experimental.constrained.fptrunc.v4f32.v4f64(<4 x double> %{{.*}}, metadata !"round.dynamic", metadata !"fpexcept.maytrap")
// SHARED: call <4 x double> @llvm.experimental.constrained.fpext.v4f64.v4f32(<4 x float> %{{.*}}, metadata !"fpexcept.maytrap")
// SHARED: call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> zeroinitializer, metadata !"une", metadata !"fpexcept.maytrap")
// SHARED: call <4 x float> @llvm.experimental.constrained.uitofp.v4f32.v4i1(<4 x i1> %{{.*}}, metadata !"round.dynamic", metadata !"fpexcept.maytrap")
