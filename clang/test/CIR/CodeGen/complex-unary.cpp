// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-canonicalize -o %t.cir %s 2>&1 | FileCheck --check-prefix=CIR-BEFORE %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-after=cir-lowering-prepare -o %t.cir %s 2>&1 | FileCheck --check-prefixes=CIR-AFTER %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

void foo() {
  int _Complex a;
  int _Complex b = ~a;
}

// CIR-BEFORE: %[[COMPLEX:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["a"]
// CIR-BEFORE: %[[RESULT:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["b", init]
// CIR-BEFORE: %[[TMP:.*]] = cir.load{{.*}} %[[COMPLEX]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR-BEFORE: %[[COMPLEX_NOT:.*]] = cir.unary(not, %[[TMP]]) : !cir.complex<!s32i>, !cir.complex<!s32i>
// CIR-BEFORE: cir.store{{.*}} %[[COMPLEX_NOT]], %[[RESULT]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>

// CIR-AFTER: %[[COMPLEX:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["a"]
// CIR-AFTER: %[[RESULT:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["b", init]
// CIR-AFTER: %[[TMP:.*]] = cir.load{{.*}} %[[COMPLEX]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR-AFTER: %[[REAL:.*]] = cir.complex.real %[[TMP]] : !cir.complex<!s32i> -> !s32i
// CIR-AFTER: %[[IMAG:.*]] = cir.complex.imag %[[TMP]] : !cir.complex<!s32i> -> !s32i
// CIR-AFTER: %[[IMAG_MINUS:.*]] = cir.unary(minus, %[[IMAG]]) : !s32i, !s32i
// CIR-AFTER: %[[RESULT_VAL:.*]] = cir.complex.create %[[REAL]], %[[IMAG_MINUS]] : !s32i -> !cir.complex<!s32i>
// CIR-AFTER: cir.store{{.*}} %[[RESULT_VAL]], %[[RESULT]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>

// LLVM: %[[COMPLEX:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[RESULT:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[TMP:.*]] = load { i32, i32 }, ptr %[[COMPLEX]], align 4
// LLVM: %[[REAL:.*]] = extractvalue { i32, i32 } %[[TMP]], 0
// LLVM: %[[IMAG:.*]] = extractvalue { i32, i32 } %[[TMP]], 1
// LLVM: %[[IMAG_MINUS:.*]] = sub i32 0, %[[IMAG]]
// LLVM: %[[RESULT_TMP:.*]] = insertvalue { i32, i32 } {{.*}}, i32 %[[REAL]], 0
// LLVM: %[[RESULT_VAL:.*]] = insertvalue { i32, i32 } %[[RESULT_TMP]], i32 %[[IMAG_MINUS]], 1
// LLVM: store { i32, i32 } %[[RESULT_VAL]], ptr %[[RESULT]], align 4

// OGCG: %[[COMPLEX:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[RESULT:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[COMPLEX]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load i32, ptr %[[A_REAL_PTR]], align 4
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[COMPLEX]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load i32, ptr %[[A_IMAG_PTR]], align 4
// OGCG: %[[A_IMAG_MINUS:.*]] = sub i32 0, %[[A_IMAG]]
// OGCG: %[[RESULT_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[RESULT]], i32 0, i32 0
// OGCG: %[[RESULT_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[RESULT]], i32 0, i32 1
// OGCG: store i32 %[[A_REAL]], ptr %[[RESULT_REAL_PTR]], align 4
// OGCG: store i32 %[[A_IMAG_MINUS]], ptr %[[RESULT_IMAG_PTR]], align 4

void foo2() {
  float _Complex a;
  float _Complex b = ~a;
}

// CIR-BEFORE: %[[COMPLEX:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR-BEFORE: %[[RESULT:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["b", init]
// CIR-BEFORE: %[[TMP:.*]] = cir.load{{.*}} %[[COMPLEX]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR-BEFORE: %[[COMPLEX_NOT:.*]] = cir.unary(not, %[[TMP]]) : !cir.complex<!cir.float>, !cir.complex<!cir.float>
// CIR-BEFORE: cir.store{{.*}} %[[COMPLEX_NOT]], %[[RESULT]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// CIR-AFTER: %[[COMPLEX:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR-AFTER: %[[RESULT:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["b", init]
// CIR-AFTER: %[[TMP:.*]] = cir.load{{.*}} %[[COMPLEX]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR-AFTER: %[[REAL:.*]] = cir.complex.real %[[TMP]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER: %[[IMAG:.*]] = cir.complex.imag %[[TMP]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER: %[[IMAG_MINUS:.*]] = cir.unary(minus, %[[IMAG]]) : !cir.float, !cir.float
// CIR-AFTER: %[[RESULT_VAL:.*]] = cir.complex.create %[[REAL]], %[[IMAG_MINUS]] : !cir.float -> !cir.complex<!cir.float>
// CIR-AFTER: cir.store{{.*}} %[[RESULT_VAL]], %[[RESULT]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// LLVM: %[[COMPLEX:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[RESULT:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[TMP:.*]] = load { float, float }, ptr %[[COMPLEX]], align 4
// LLVM: %[[REAL:.*]] = extractvalue { float, float } %[[TMP]], 0
// LLVM: %[[IMAG:.*]] = extractvalue { float, float } %[[TMP]], 1
// LLVM: %[[IMAG_MINUS:.*]] = fneg float %[[IMAG]]
// LLVM: %[[RESULT_TMP:.*]] = insertvalue { float, float } {{.*}}, float %[[REAL]], 0
// LLVM: %[[RESULT_VAL:.*]] = insertvalue { float, float } %[[RESULT_TMP]], float %[[IMAG_MINUS]], 1
// LLVM: store { float, float } %[[RESULT_VAL]], ptr %[[RESULT]], align 4

// OGCG: %[[COMPLEX:.*]] = alloca { float, float }, align 4
// OGCG: %[[RESULT:.*]] = alloca { float, float }, align 4
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[COMPLEX]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load float, ptr %[[A_REAL_PTR]], align 4
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[COMPLEX]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load float, ptr %[[A_IMAG_PTR]], align 4
// OGCG: %[[A_IMAG_MINUS:.*]] = fneg float  %[[A_IMAG]]
// OGCG: %[[RESULT_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[RESULT]], i32 0, i32 0
// OGCG: %[[RESULT_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[RESULT]], i32 0, i32 1
// OGCG: store float %[[A_REAL]], ptr %[[RESULT_REAL_PTR]], align 4
// OGCG: store float %[[A_IMAG_MINUS]], ptr %[[RESULT_IMAG_PTR]], align 4
