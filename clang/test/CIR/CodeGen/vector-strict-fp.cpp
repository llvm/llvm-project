// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -ffp-contract=on -fexperimental-strict-floating-point -ffp-exception-behavior=strict -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -ffp-contract=on -fexperimental-strict-floating-point -ffp-exception-behavior=strict -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -ffp-contract=on -fexperimental-strict-floating-point -ffp-exception-behavior=strict -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

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

// LLVM: %[[A_ADDR:.*]] = alloca <4 x float>, align 16
// LLVM: %[[B_ADDR:.*]] = alloca <4 x i32>, align 16
// LLVM: %[[TMP_A:.*]] = load <4 x float>, ptr %[[A_ADDR]], align 16
// LLVM: %[[RESULT:.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %[[TMP_A]], <4 x float> zeroinitializer, metadata !"oeq", metadata !"fpexcept.strict")
// LLVM: %[[RESULT_V4I:.*]] = sext <4 x i1> %[[RESULT]] to <4 x i32>
// LLVM: store <4 x i32> %[[RESULT_V4I]], ptr %[[B_ADDR]], align 16
