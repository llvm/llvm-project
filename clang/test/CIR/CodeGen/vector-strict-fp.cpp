// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -ffp-contract=on -fexperimental-strict-floating-point -ffp-exception-behavior=strict -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -ffp-contract=on -fexperimental-strict-floating-point -ffp-exception-behavior=strict -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefixes=LLVM,SHARED
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -ffp-contract=on -fexperimental-strict-floating-point -ffp-exception-behavior=strict -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefixes=OGCG,SHARED

typedef float v4f __attribute__((ext_vector_type(4)));
typedef int v4i __attribute__((ext_vector_type(4)));

void vec_logical_not() {
  v4f a;
  v4i b = !a;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!cir.vector<4 x !cir.float>>
// CIR: %[[B_ADDR:.*]] = cir.alloca "b" {{.*}} init : !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float>
// CIR: %[[CONST_ZERO:.*]] = cir.const #cir.zero : !cir.vector<4 x !cir.float>
// CIR: %[[RESULT:.*]] = cir.vec.cmp(eq, %[[TMP_A]], %[[CONST_ZERO]]) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !s32i> {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
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
// CIR: %[[A_NE_ZERO:.*]] = cir.vec.cmp(ne, %[[TMP_A]], %[[CONST_ZERO]]) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !s32i> {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR: %[[B_NE_ZERO:.*]] = cir.vec.cmp(ne, %[[TMP_B]], %[[CONST_ZERO]]) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !s32i> {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR: %[[RESULT:.*]] = cir.or %[[A_NE_ZERO]], %[[B_NE_ZERO]] : !cir.vector<4 x !s32i>
// CIR: cir.store {{.*}} %[[RESULT]], %[[R_ADDR]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>

// SHARED: %[[A_ADDR:.*]] = alloca <4 x float>, align 16
// SHARED: %[[B_ADDR:.*]] = alloca <4 x float>, align 16
// SHARED: %[[R_ADDR:.*]] = alloca <4 x i32>, align 16
// SHARED: %[[TMP_A:.*]] = load <4 x float>, ptr %[[A_ADDR]], align 16
// SHARED: %[[TMP_B:.*]] = load <4 x float>, ptr %[[B_ADDR]], align 16

// In CIR IR vec compare will perform sext cast -> logical or

// LLVM: %[[A_NE_ZERO:.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %[[TMP_A]], <4 x float> zeroinitializer, metadata !"une", metadata !"fpexcept.strict")
// LLVM: %[[A_NE_ZERO_VEC:.*]] = sext <4 x i1> %[[A_NE_ZERO]] to <4 x i32>
// LLVM: %[[B_NE_ZERO:.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %[[TMP_B]], <4 x float> zeroinitializer, metadata !"une", metadata !"fpexcept.strict")
// LLVM: %[[B_NE_ZERO_VEC:.*]] = sext <4 x i1> %[[B_NE_ZERO]] to <4 x i32>
// LLVM: %[[RESULT_VEC:.*]] = or <4 x i32> %[[A_NE_ZERO_VEC]], %[[B_NE_ZERO_VEC]]

// In OG it's perform compare -> logical or -> sext

// OGCG: %[[A_NE_ZERO:.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %[[TMP_A]], <4 x float> zeroinitializer, metadata !"une", metadata !"fpexcept.strict")
// OGCG: %[[B_NE_ZERO:.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %[[TMP_B]], <4 x float> zeroinitializer, metadata !"une", metadata !"fpexcept.strict")
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
// CIR: %[[A_NE_ZERO:.*]] = cir.vec.cmp(ne, %[[TMP_A]], %[[CONST_ZERO]]) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !s32i> {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR: %[[B_NE_ZERO:.*]] = cir.vec.cmp(ne, %[[TMP_B]], %[[CONST_ZERO]]) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !s32i> {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR: %[[RESULT:.*]] = cir.and %[[A_NE_ZERO]], %[[B_NE_ZERO]] : !cir.vector<4 x !s32i>
// CIR: cir.store {{.*}} %[[RESULT]], %[[R_ADDR]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>

// SHARED: %[[A_ADDR:.*]] = alloca <4 x float>, align 16
// SHARED: %[[B_ADDR:.*]] = alloca <4 x float>, align 16
// SHARED: %[[R_ADDR:.*]] = alloca <4 x i32>, align 16
// SHARED: %[[TMP_A:.*]] = load <4 x float>, ptr %[[A_ADDR]], align 16
// SHARED: %[[TMP_B:.*]] = load <4 x float>, ptr %[[B_ADDR]], align 16

// In CIR IR vec compare will perform sext cast -> logical and

// LLVM: %[[A_NE_ZERO:.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %[[TMP_A]], <4 x float> zeroinitializer, metadata !"une", metadata !"fpexcept.strict")
// LLVM: %[[A_NE_ZERO_VEC:.*]] = sext <4 x i1> %[[A_NE_ZERO]] to <4 x i32>
// LLVM: %[[B_NE_ZERO:.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %[[TMP_B]], <4 x float> zeroinitializer, metadata !"une", metadata !"fpexcept.strict")
// LLVM: %[[B_NE_ZERO_VEC:.*]] = sext <4 x i1> %[[B_NE_ZERO]] to <4 x i32>
// LLVM: %[[RESULT_VEC:.*]] = and <4 x i32> %[[A_NE_ZERO_VEC]], %[[B_NE_ZERO_VEC]]

// In OG it's perform compare -> logical and -> sext

// OGCG: %[[A_NE_ZERO:.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %[[TMP_A]], <4 x float> zeroinitializer, metadata !"une", metadata !"fpexcept.strict") #2
// OGCG: %[[B_NE_ZERO:.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %[[TMP_B]], <4 x float> zeroinitializer, metadata !"une", metadata !"fpexcept.strict") #2
// OGCG: %[[RESULT:.*]] = and <4 x i1> %[[A_NE_ZERO]], %[[B_NE_ZERO]]
// OGCG: %[[RESULT_VEC:.*]] = sext <4 x i1> %2 to <4 x i32>

// SHARED: store <4 x i32> %[[RESULT_VEC]], ptr %[[R_ADDR]], align 16
