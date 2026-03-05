// RUN: %clang_cc1 -x c -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=C_CIR

// RUN: %clang_cc1 -x c -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=C_LLVM

// RUN: %clang_cc1 -x c -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=C_OGCG

#ifndef __cplusplus
void foo() {
  float _Complex a;
  float b;
  b += a;
}
#endif

// C_CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// C_CIR: %[[B_ADDR:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["b"]
// C_CIR: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// C_CIR: %[[TMP_B:.*]] = cir.load{{.*}} %[[B_ADDR]] : !cir.ptr<!cir.float>, !cir.float
// C_CIR: %[[CONST_ZERO:.*]] = cir.const #cir.fp<0.000000e+00> : !cir.float
// C_CIR: %[[COMPLEX_B:.*]] = cir.complex.create %[[TMP_B]], %[[CONST_ZERO]] : !cir.float -> !cir.complex<!cir.float>
// C_CIR: %[[B_REAL:.*]] = cir.complex.real %[[COMPLEX_B]] : !cir.complex<!cir.float> -> !cir.float
// C_CIR: %[[B_IMAG:.*]] = cir.complex.imag %[[COMPLEX_B]] : !cir.complex<!cir.float> -> !cir.float
// C_CIR: %[[A_REAL:.*]] = cir.complex.real %[[TMP_A]] : !cir.complex<!cir.float> -> !cir.float
// C_CIR: %[[A_IMAG:.*]] = cir.complex.imag %[[TMP_A]] : !cir.complex<!cir.float> -> !cir.float
// C_CIR: %[[ADD_REAL:.*]] = cir.binop(add, %[[B_REAL]], %[[A_REAL]]) : !cir.float
// C_CIR: %[[ADD_IMAG:.*]] = cir.binop(add, %[[B_IMAG]], %[[A_IMAG]]) : !cir.float
// C_CIR: %[[RESULT_COMPLEX:.*]] = cir.complex.create %[[ADD_REAL]], %[[ADD_IMAG]] : !cir.float -> !cir.complex<!cir.float>
// C_CIR: %[[RESULT_REAL:.*]] = cir.complex.real %[[RESULT_COMPLEX]] : !cir.complex<!cir.float> -> !cir.float
// C_CIR: cir.store{{.*}} %[[RESULT_REAL]], %[[B_ADDR]] : !cir.float, !cir.ptr<!cir.float>

// C_LLVM: %[[A_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// C_LLVM: %[[B_ADDR:.*]] = alloca float, i64 1, align 4
// C_LLVM: %[[TMP_A:.*]] = load { float, float }, ptr %[[A_ADDR]], align 4
// C_LLVM: %[[TMP_B:.*]] = load float, ptr %[[B_ADDR]], align 4
// C_LLVM: %[[TMP_COMPLEX_B:.*]] = insertvalue { float, float } {{.*}}, float %[[TMP_B]], 0
// C_LLVM: %[[COMPLEX_B:.*]] = insertvalue { float, float } %[[TMP_COMPLEX_B]], float 0.000000e+00, 1
// C_LLVM: %[[A_REAL:.*]] = extractvalue { float, float } %[[TMP_A]], 0
// C_LLVM: %[[A_IMAG:.*]] = extractvalue { float, float } %[[TMP_A]], 1
// C_LLVM: %[[RESULT_REAL:.*]] = fadd float %[[TMP_B]], %[[A_REAL]]
// C_LLVM: %[[RESULT_IMAG:.*]] = fadd float 0.000000e+00, %[[A_IMAG]]
// C_LLVM: %[[TMP_RESULT:.*]] = insertvalue { float, float } {{.*}}, float %[[RESULT_REAL]], 0
// C_LLVM: %[[RESULT:.*]] = insertvalue { float, float } %[[TMP_RESULT]], float %[[RESULT_IMAG]], 1
// C_LLVM: store float %[[RESULT_REAL]], ptr %[[B_ADDR]], align 4

// C_OGCG: %[[A_ADDR:.*]] = alloca { float, float }, align 4
// C_OGCG: %[[B_ADDR:.*]] = alloca float, align 4
// C_OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// C_OGCG: %[[A_REAL:.*]] = load float, ptr %[[A_REAL_PTR]], align 4
// C_OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// C_OGCG: %[[A_IMAG:.*]] = load float, ptr %[[A_IMAG_PTR]], align 4
// C_OGCG: %[[TMP_B:.*]] = load float, ptr %[[B_ADDR]], align 4
// C_OGCG: %[[ADD_REAL:.*]] = fadd float %[[TMP_B]], %[[A_REAL]]
// C_OGCG: store float %[[ADD_REAL]], ptr %[[B_ADDR]], align 4
