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

// CIR-BEFORE: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["a"]
// CIR-BEFORE: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["b", init]
// CIR-BEFORE: %[[TMP:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR-BEFORE: %[[COMPLEX_NOT:.*]] = cir.unary(not, %[[TMP]]) : !cir.complex<!s32i>, !cir.complex<!s32i>
// CIR-BEFORE: cir.store{{.*}} %[[COMPLEX_NOT]], %[[B_ADDR]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>

// CIR-AFTER: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["a"]
// CIR-AFTER: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["b", init]
// CIR-AFTER: %[[TMP:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR-AFTER: %[[REAL:.*]] = cir.complex.real %[[TMP]] : !cir.complex<!s32i> -> !s32i
// CIR-AFTER: %[[IMAG:.*]] = cir.complex.imag %[[TMP]] : !cir.complex<!s32i> -> !s32i
// CIR-AFTER: %[[IMAG_MINUS:.*]] = cir.unary(minus, %[[IMAG]]) : !s32i, !s32i
// CIR-AFTER: %[[RESULT_VAL:.*]] = cir.complex.create %[[REAL]], %[[IMAG_MINUS]] : !s32i -> !cir.complex<!s32i>
// CIR-AFTER: cir.store{{.*}} %[[RESULT_VAL]], %[[B_ADDR]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>

// LLVM: %[[A_ADDR:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[B_ADDR:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[TMP:.*]] = load { i32, i32 }, ptr %[[A_ADDR]], align 4
// LLVM: %[[REAL:.*]] = extractvalue { i32, i32 } %[[TMP]], 0
// LLVM: %[[IMAG:.*]] = extractvalue { i32, i32 } %[[TMP]], 1
// LLVM: %[[IMAG_MINUS:.*]] = sub i32 0, %[[IMAG]]
// LLVM: %[[RESULT_TMP:.*]] = insertvalue { i32, i32 } {{.*}}, i32 %[[REAL]], 0
// LLVM: %[[RESULT_VAL:.*]] = insertvalue { i32, i32 } %[[RESULT_TMP]], i32 %[[IMAG_MINUS]], 1
// LLVM: store { i32, i32 } %[[RESULT_VAL]], ptr %[[B_ADDR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[B_ADDR:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load i32, ptr %[[A_REAL_PTR]], align 4
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load i32, ptr %[[A_IMAG_PTR]], align 4
// OGCG: %[[A_IMAG_MINUS:.*]] = sub i32 0, %[[A_IMAG]]
// OGCG: %[[RESULT_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG: %[[RESULT_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG: store i32 %[[A_REAL]], ptr %[[RESULT_REAL_PTR]], align 4
// OGCG: store i32 %[[A_IMAG_MINUS]], ptr %[[RESULT_IMAG_PTR]], align 4

void foo2() {
  float _Complex a;
  float _Complex b = ~a;
}

// CIR-BEFORE: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR-BEFORE: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["b", init]
// CIR-BEFORE: %[[TMP:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR-BEFORE: %[[COMPLEX_NOT:.*]] = cir.unary(not, %[[TMP]]) : !cir.complex<!cir.float>, !cir.complex<!cir.float>
// CIR-BEFORE: cir.store{{.*}} %[[COMPLEX_NOT]], %[[B_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// CIR-AFTER: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR-AFTER: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["b", init]
// CIR-AFTER: %[[TMP:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR-AFTER: %[[REAL:.*]] = cir.complex.real %[[TMP]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER: %[[IMAG:.*]] = cir.complex.imag %[[TMP]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER: %[[IMAG_MINUS:.*]] = cir.unary(minus, %[[IMAG]]) : !cir.float, !cir.float
// CIR-AFTER: %[[RESULT_VAL:.*]] = cir.complex.create %[[REAL]], %[[IMAG_MINUS]] : !cir.float -> !cir.complex<!cir.float>
// CIR-AFTER: cir.store{{.*}} %[[RESULT_VAL]], %[[B_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// LLVM: %[[A_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[B_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[TMP:.*]] = load { float, float }, ptr %[[A_ADDR]], align 4
// LLVM: %[[REAL:.*]] = extractvalue { float, float } %[[TMP]], 0
// LLVM: %[[IMAG:.*]] = extractvalue { float, float } %[[TMP]], 1
// LLVM: %[[IMAG_MINUS:.*]] = fneg float %[[IMAG]]
// LLVM: %[[RESULT_TMP:.*]] = insertvalue { float, float } {{.*}}, float %[[REAL]], 0
// LLVM: %[[RESULT_VAL:.*]] = insertvalue { float, float } %[[RESULT_TMP]], float %[[IMAG_MINUS]], 1
// LLVM: store { float, float } %[[RESULT_VAL]], ptr %[[B_ADDR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[B_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load float, ptr %[[A_REAL_PTR]], align 4
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load float, ptr %[[A_IMAG_PTR]], align 4
// OGCG: %[[A_IMAG_MINUS:.*]] = fneg float %[[A_IMAG]]
// OGCG: %[[RESULT_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG: %[[RESULT_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG: store float %[[A_REAL]], ptr %[[RESULT_REAL_PTR]], align 4
// OGCG: store float %[[A_IMAG_MINUS]], ptr %[[RESULT_IMAG_PTR]], align 4

void foo3() {
  float _Complex a;
  float _Complex b = a++;
}

// CIR-BEFORE: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR-BEFORE: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["b", init]
// CIR-BEFORE: %[[TMP:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR-BEFORE: %[[COMPLEX_INC:.*]] = cir.unary(inc, %[[TMP]]) : !cir.complex<!cir.float>, !cir.complex<!cir.float>
// CIR-BEFORE: cir.store{{.*}} %[[COMPLEX_INC]], %[[A_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>
// CIR-BEFORE: cir.store{{.*}} %[[TMP]], %[[B_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// CIR-AFTER: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR-AFTER: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["b", init]
// CIR-AFTER: %[[TMP:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR-AFTER: %[[REAL:.*]] = cir.complex.real %[[TMP]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER: %[[IMAG:.*]] = cir.complex.imag %[[TMP]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER: %[[REAL_INC:.*]] = cir.unary(inc, %[[REAL]]) : !cir.float, !cir.float
// CIR-AFTER: %[[NEW_COMPLEX:.*]] = cir.complex.create %[[REAL_INC]], %[[IMAG]] : !cir.float -> !cir.complex<!cir.float>
// CIR-AFTER: cir.store{{.*}} %[[NEW_COMPLEX]], %[[A_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>
// CIR-AFTER: cir.store{{.*}} %[[TMP]], %[[B_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// LLVM: %[[A_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[B_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[TMP:.*]] = load { float, float }, ptr %[[A_ADDR]], align 4
// LLVM: %[[REAL:.*]] = extractvalue { float, float } %[[TMP]], 0
// LLVM: %[[IMAG:.*]] = extractvalue { float, float } %[[TMP]], 1
// LLVM: %[[REAL_INC:.*]] = fadd float 1.000000e+00, %[[REAL]]
// LLVM: %[[RESULT_TMP:.*]] = insertvalue { float, float } {{.*}}, float %[[REAL_INC]], 0
// LLVM: %[[RESULT_VAL:.*]] = insertvalue { float, float } %[[RESULT_TMP]], float %[[IMAG]], 1
// LLVM: store { float, float } %[[RESULT_VAL]], ptr %[[A_ADDR]], align 4
// LLVM: store { float, float } %[[TMP]], ptr %[[B_ADDR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[B_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load float, ptr %[[A_REAL_PTR]], align 4
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load float, ptr %[[A_IMAG_PTR]], align 4
// OGCG: %[[A_REAL_INC:.*]] = fadd float %[[A_REAL]], 1.000000e+00
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: store float %[[A_REAL_INC]], ptr %[[A_REAL_PTR]], align 4
// OGCG: store float %[[A_IMAG]], ptr %[[A_IMAG_PTR]], align 4
// OGCG: %[[RESULT_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG: %[[RESULT_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG: store float %[[A_REAL]], ptr %[[RESULT_REAL_PTR]], align 4
// OGCG: store float %[[A_IMAG]], ptr %[[RESULT_IMAG_PTR]], align 4

void foo4() {
  float _Complex a;
  float _Complex b = ++a;
}

// CIR-BEFORE: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR-BEFORE: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["b", init]
// CIR-BEFORE: %[[TMP:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR-BEFORE: %[[COMPLEX_INC:.*]] = cir.unary(inc, %[[TMP]]) : !cir.complex<!cir.float>, !cir.complex<!cir.float>
// CIR-BEFORE: cir.store{{.*}} %[[COMPLEX_INC]], %[[A_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>
// CIR-BEFORE: cir.store{{.*}} %[[COMPLEX_INC]], %[[B_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// CIR-AFTER: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR-AFTER: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["b", init]
// CIR-AFTER: %[[TMP:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR-AFTER: %[[REAL:.*]] = cir.complex.real %[[TMP]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER: %[[IMAG:.*]] = cir.complex.imag %[[TMP]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER: %[[REAL_INC:.*]] = cir.unary(inc, %[[REAL]]) : !cir.float, !cir.float
// CIR-AFTER: %[[NEW_COMPLEX:.*]] = cir.complex.create %[[REAL_INC]], %[[IMAG]] : !cir.float -> !cir.complex<!cir.float>
// CIR-AFTER: cir.store{{.*}} %[[NEW_COMPLEX]], %[[A_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>
// CIR-AFTER: cir.store{{.*}} %[[NEW_COMPLEX]], %[[B_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// LLVM: %[[A_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[B_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[TMP:.*]] = load { float, float }, ptr %[[A_ADDR]], align 4
// LLVM: %[[REAL:.*]] = extractvalue { float, float } %[[TMP]], 0
// LLVM: %[[IMAG:.*]] = extractvalue { float, float } %[[TMP]], 1
// LLVM: %[[REAL_INC:.*]] = fadd float 1.000000e+00, %[[REAL]]
// LLVM: %[[RESULT_TMP:.*]] = insertvalue { float, float } {{.*}}, float %[[REAL_INC]], 0
// LLVM: %[[RESULT_VAL:.*]] = insertvalue { float, float } %[[RESULT_TMP]], float %[[IMAG]], 1
// LLVM: store { float, float } %[[RESULT_VAL]], ptr %[[A_ADDR]], align 4
// LLVM: store { float, float } %[[RESULT_VAL]], ptr %[[B_ADDR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[B_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load float, ptr %[[A_REAL_PTR]], align 4
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load float, ptr %[[A_IMAG_PTR]], align 4
// OGCG: %[[A_REAL_INC:.*]] = fadd float %[[A_REAL]], 1.000000e+00
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: store float %[[A_REAL_INC]], ptr %[[A_REAL_PTR]], align 4
// OGCG: store float %[[A_IMAG]], ptr %[[A_IMAG_PTR]], align 4
// OGCG: %[[RESULT_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG: %[[RESULT_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG: store float %[[A_REAL_INC]], ptr %[[RESULT_REAL_PTR]], align 4
// OGCG: store float %[[A_IMAG]], ptr %[[RESULT_IMAG_PTR]], align 4

void foo5() {
  float _Complex a;
  float _Complex b = a--;
}

// CIR-BEFORE: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR-BEFORE: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["b", init]
// CIR-BEFORE: %[[TMP:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR-BEFORE: %[[COMPLEX_DEC:.*]] = cir.unary(dec, %[[TMP]]) : !cir.complex<!cir.float>, !cir.complex<!cir.float>
// CIR-BEFORE: cir.store{{.*}} %[[COMPLEX_DEC]], %[[A_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>
// CIR-BEFORE: cir.store{{.*}} %[[TMP]], %[[B_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// CIR-AFTER: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR-AFTER: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["b", init]
// CIR-AFTER: %[[TMP:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR-AFTER: %[[REAL:.*]] = cir.complex.real %[[TMP]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER: %[[IMAG:.*]] = cir.complex.imag %[[TMP]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER: %[[REAL_DEC:.*]] = cir.unary(dec, %[[REAL]]) : !cir.float, !cir.float
// CIR-AFTER: %[[NEW_COMPLEX:.*]] = cir.complex.create %[[REAL_DEC]], %[[IMAG]] : !cir.float -> !cir.complex<!cir.float>
// CIR-AFTER: cir.store{{.*}} %[[NEW_COMPLEX]], %[[A_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>
// CIR-AFTER: cir.store{{.*}} %[[TMP]], %[[B_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// LLVM: %[[A_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[B_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[TMP:.*]] = load { float, float }, ptr %[[A_ADDR]], align 4
// LLVM: %[[REAL:.*]] = extractvalue { float, float } %[[TMP]], 0
// LLVM: %[[IMAG:.*]] = extractvalue { float, float } %[[TMP]], 1
// LLVM: %[[REAL_DEC:.*]] = fadd float -1.000000e+00, %[[REAL]]
// LLVM: %[[RESULT_TMP:.*]] = insertvalue { float, float } {{.*}}, float %[[REAL_DEC]], 0
// LLVM: %[[RESULT_VAL:.*]] = insertvalue { float, float } %[[RESULT_TMP]], float %[[IMAG]], 1
// LLVM: store { float, float } %[[RESULT_VAL]], ptr %[[A_ADDR]], align 4
// LLVM: store { float, float } %[[TMP]], ptr %[[B_ADDR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[B_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load float, ptr %[[A_REAL_PTR]], align 4
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load float, ptr %[[A_IMAG_PTR]], align 4
// OGCG: %[[A_REAL_DEC:.*]] = fadd float %[[A_REAL]], -1.000000e+00
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: store float %[[A_REAL_DEC]], ptr %[[A_REAL_PTR]], align 4
// OGCG: store float %[[A_IMAG]], ptr %[[A_IMAG_PTR]], align 4
// OGCG: %[[RESULT_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG: %[[RESULT_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG: store float %[[A_REAL]], ptr %[[RESULT_REAL_PTR]], align 4
// OGCG: store float %[[A_IMAG]], ptr %[[RESULT_IMAG_PTR]], align 4

void foo6() {
  float _Complex a;
  float _Complex b = --a;
}

// CIR-BEFORE: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR-BEFORE: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["b", init]
// CIR-BEFORE: %[[TMP:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR-BEFORE: %[[COMPLEX_DEC:.*]] = cir.unary(dec, %[[TMP]]) : !cir.complex<!cir.float>, !cir.complex<!cir.float>
// CIR-BEFORE: cir.store{{.*}} %[[COMPLEX_DEC]], %[[A_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>
// CIR-BEFORE: cir.store{{.*}} %[[COMPLEX_DEC]], %[[B_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// CIR-AFTER: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR-AFTER: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["b", init]
// CIR-AFTER: %[[TMP:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR-AFTER: %[[REAL:.*]] = cir.complex.real %[[TMP]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER: %[[IMAG:.*]] = cir.complex.imag %[[TMP]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER: %[[REAL_DEC:.*]] = cir.unary(dec, %[[REAL]]) : !cir.float, !cir.float
// CIR-AFTER: %[[NEW_COMPLEX:.*]] = cir.complex.create %[[REAL_DEC]], %[[IMAG]] : !cir.float -> !cir.complex<!cir.float>
// CIR-AFTER: cir.store{{.*}} %[[NEW_COMPLEX]], %[[A_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>
// CIR-AFTER: cir.store{{.*}} %[[NEW_COMPLEX]], %[[B_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// LLVM: %[[A_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[B_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[TMP:.*]] = load { float, float }, ptr %[[A_ADDR]], align 4
// LLVM: %[[REAL:.*]] = extractvalue { float, float } %[[TMP]], 0
// LLVM: %[[IMAG:.*]] = extractvalue { float, float } %[[TMP]], 1
// LLVM: %[[REAL_DEC:.*]] = fadd float -1.000000e+00, %[[REAL]]
// LLVM: %[[RESULT_TMP:.*]] = insertvalue { float, float } {{.*}}, float %[[REAL_DEC]], 0
// LLVM: %[[RESULT_VAL:.*]] = insertvalue { float, float } %[[RESULT_TMP]], float %[[IMAG]], 1
// LLVM: store { float, float } %[[RESULT_VAL]], ptr %[[A_ADDR]], align 4
// LLVM: store { float, float } %[[RESULT_VAL]], ptr %[[B_ADDR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[B_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load float, ptr %[[A_REAL_PTR]], align 4
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load float, ptr %[[A_IMAG_PTR]], align 4
// OGCG: %[[A_REAL_DEC:.*]] = fadd float %[[A_REAL]], -1.000000e+00
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: store float %[[A_REAL_DEC]], ptr %[[A_REAL_PTR]], align 4
// OGCG: store float %[[A_IMAG]], ptr %[[A_IMAG_PTR]], align 4
// OGCG: %[[RESULT_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG: %[[RESULT_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG: store float %[[A_REAL_DEC]], ptr %[[RESULT_REAL_PTR]], align 4
// OGCG: store float %[[A_IMAG]], ptr %[[RESULT_IMAG_PTR]], align 4

void foo7() {
  float _Complex a;
  float _Complex b = +a;
}

// CIR-BEFORE: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR-BEFORE: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["b", init]
// CIR-BEFORE: %[[TMP:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR-BEFORE: %[[COMPLEX_PLUS:.*]] = cir.unary(plus, %[[TMP]]) : !cir.complex<!cir.float>, !cir.complex<!cir.float>
// CIR-BEFORE: cir.store{{.*}} %[[COMPLEX_PLUS]], %[[B_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// CIR-AFTER: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR-AFTER: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["b", init]
// CIR-AFTER: %[[TMP:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR-AFTER: %[[REAL:.*]] = cir.complex.real %[[TMP]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER: %[[IMAG:.*]] = cir.complex.imag %[[TMP]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER: %[[REAL_PLUS:.*]] = cir.unary(plus, %[[REAL]]) : !cir.float, !cir.float
// CIR-AFTER: %[[IMAG_PLUS:.*]] = cir.unary(plus, %[[IMAG]]) : !cir.float, !cir.float
// CIR-AFTER: %[[NEW_COMPLEX:.*]] = cir.complex.create %[[REAL_PLUS]], %[[IMAG_PLUS]] : !cir.float -> !cir.complex<!cir.float>
// CIR-AFTER: cir.store{{.*}} %[[NEW_COMPLEX]], %[[B_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// LLVM: %[[A_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[B_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[TMP:.*]] = load { float, float }, ptr %[[A_ADDR]], align 4
// LLVM: %[[REAL:.*]] = extractvalue { float, float } %[[TMP]], 0
// LLVM: %[[IMAG:.*]] = extractvalue { float, float } %[[TMP]], 1
// LLVM: %[[RESULT_TMP:.*]] = insertvalue { float, float } {{.*}}, float %[[REAL]], 0
// LLVM: %[[RESULT_VAL:.*]] = insertvalue { float, float } %[[RESULT_TMP]], float %[[IMAG]], 1
// LLVM: store { float, float } %[[RESULT_VAL]], ptr %[[B_ADDR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[B_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load float, ptr %[[A_REAL_PTR]], align 4
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load float, ptr %[[A_IMAG_PTR]], align 4
// OGCG: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG: store float %[[A_REAL]], ptr %[[B_REAL_PTR]], align 4
// OGCG: store float %[[A_IMAG]], ptr %[[B_IMAG_PTR]], align 4

void foo8() {
  float _Complex a;
  float _Complex b = -a;
}

// CIR-BEFORE: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR-BEFORE: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["b", init]
// CIR-BEFORE: %[[TMP:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR-BEFORE: %[[COMPLEX_MINUS:.*]] = cir.unary(minus, %[[TMP]]) : !cir.complex<!cir.float>, !cir.complex<!cir.float>
// CIR-BEFORE: cir.store{{.*}} %[[COMPLEX_MINUS]], %[[B_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// CIR-AFTER: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR-AFTER: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["b", init]
// CIR-AFTER: %[[TMP:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR-AFTER: %[[REAL:.*]] = cir.complex.real %[[TMP]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER: %[[IMAG:.*]] = cir.complex.imag %[[TMP]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER: %[[REAL_MINUS:.*]] = cir.unary(minus, %[[REAL]]) : !cir.float, !cir.float
// CIR-AFTER: %[[IMAG_MINUS:.*]] = cir.unary(minus, %[[IMAG]]) : !cir.float, !cir.float
// CIR-AFTER: %[[NEW_COMPLEX:.*]] = cir.complex.create %[[REAL_MINUS]], %[[IMAG_MINUS]] : !cir.float -> !cir.complex<!cir.float>
// CIR-AFTER: cir.store{{.*}} %[[NEW_COMPLEX]], %[[B_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// LLVM: %[[A_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[B_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[TMP:.*]] = load { float, float }, ptr %[[A_ADDR]], align 4
// LLVM: %[[REAL:.*]] = extractvalue { float, float } %[[TMP]], 0
// LLVM: %[[IMAG:.*]] = extractvalue { float, float } %[[TMP]], 1
// LLVM: %[[REAL_MINUS:.*]] = fneg float %[[REAL]]
// LLVM: %[[IMAG_MINUS:.*]] = fneg float %[[IMAG]]
// LLVM: %[[RESULT_TMP:.*]] = insertvalue { float, float } {{.*}}, float %[[REAL_MINUS]], 0
// LLVM: %[[RESULT_VAL:.*]] = insertvalue { float, float } %[[RESULT_TMP]], float %[[IMAG_MINUS]], 1
// LLVM: store { float, float } %[[RESULT_VAL]], ptr %[[B_ADDR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[B_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load float, ptr %[[A_REAL_PTR]], align 4
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load float, ptr %[[A_IMAG_PTR]], align 4
// OGCG: %[[A_REAL_MINUS:.*]] = fneg float %[[A_REAL]]
// OGCG: %[[A_IMAG_MINUS:.*]] = fneg float %[[A_IMAG]]
// OGCG: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG: store float %[[A_REAL_MINUS]], ptr %[[B_REAL_PTR]], align 4
// OGCG: store float %[[A_IMAG_MINUS]], ptr %[[B_IMAG_PTR]], align 4

void foo9() {
  _Float16 _Complex a;
  _Float16 _Complex b = +a;
}


// CIR-BEFORE: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.f16>, !cir.ptr<!cir.complex<!cir.f16>>, ["a"]
// CIR-BEFORE: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!cir.f16>, !cir.ptr<!cir.complex<!cir.f16>>, ["b", init]
// CIR-BEFORE: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.f16>>, !cir.complex<!cir.f16>
// CIR-BEFORE: %[[A_COMPLEX_F32:.*]] = cir.cast(float_complex, %[[TMP_A]] : !cir.complex<!cir.f16>), !cir.complex<!cir.float>
// CIR-BEFORE: %[[RESULT:.*]] = cir.unary(plus, %[[A_COMPLEX_F32]]) : !cir.complex<!cir.float>, !cir.complex<!cir.float>
// CIR-BEFORE: %[[A_COMPLEX_F16:.*]] = cir.cast(float_complex, %[[RESULT]] : !cir.complex<!cir.float>), !cir.complex<!cir.f16>
// CIR-BEFORE: cir.store{{.*}} %[[A_COMPLEX_F16]], %[[B_ADDR]] : !cir.complex<!cir.f16>, !cir.ptr<!cir.complex<!cir.f16>>

// CIR-AFTER: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.f16>, !cir.ptr<!cir.complex<!cir.f16>>, ["a"]
// CIR-AFTER: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!cir.f16>, !cir.ptr<!cir.complex<!cir.f16>>, ["b", init]
// CIR-AFTER: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.f16>>, !cir.complex<!cir.f16>
// CIR-AFTER: %[[A_REAL:.*]] = cir.complex.real %[[TMP_A]] : !cir.complex<!cir.f16> -> !cir.f16
// CIR-AFTER: %[[A_IMAG:.*]] = cir.complex.imag %[[TMP_A]] : !cir.complex<!cir.f16> -> !cir.f16
// CIR-AFTER: %[[A_REAL_F32:.*]] = cir.cast(floating, %[[A_REAL]] : !cir.f16), !cir.float
// CIR-AFTER: %[[A_IMAG_F32:.*]] = cir.cast(floating, %[[A_IMAG]] : !cir.f16), !cir.float
// CIR-AFTER: %[[A_COMPLEX_F32:.*]] = cir.complex.create %[[A_REAL_F32]], %[[A_IMAG_F32]] : !cir.float -> !cir.complex<!cir.float>
// CIR-AFTER: %[[A_REAL_F32:.*]] = cir.complex.real %[[A_COMPLEX_F32]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER: %[[A_IMAG_F32:.*]] = cir.complex.imag %[[A_COMPLEX_F32]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER: %[[RESULT_REAL_F32:.*]] = cir.unary(plus, %[[A_REAL_F32]]) : !cir.float, !cir.float
// CIR-AFTER: %[[RESULT_IMAG_F32:.*]] = cir.unary(plus, %[[A_IMAG_F32]]) : !cir.float, !cir.float
// CIR-AFTER: %[[RESULT_COMPLEX_F32:.*]] = cir.complex.create %[[RESULT_REAL_F32]], %[[RESULT_IMAG_F32]] : !cir.float -> !cir.complex<!cir.float>
// CIR-AFTER: %[[RESULT_REAL_F32:.*]] = cir.complex.real %[[RESULT_COMPLEX_F32]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER: %[[RESULT_IMAG_F32:.*]] = cir.complex.imag %[[RESULT_COMPLEX_F32]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER: %[[RESULT_REAL_F16:.*]] = cir.cast(floating, %[[RESULT_REAL_F32]] : !cir.float), !cir.f16
// CIR-AFTER: %[[RESULT_IMAG_F16:.*]] = cir.cast(floating, %[[RESULT_IMAG_F32]] : !cir.float), !cir.f16
// CIR-AFTER: %[[RESULT_COMPLEX_F16:.*]] = cir.complex.create %[[RESULT_REAL_F16]], %[[RESULT_IMAG_F16]] : !cir.f16 -> !cir.complex<!cir.f16>
// CIR-AFTER: cir.store{{.*}} %[[RESULT_COMPLEX_F16]], %[[B_ADDR]] : !cir.complex<!cir.f16>, !cir.ptr<!cir.complex<!cir.f16>>

// LLVM: %[[A_ADDR:.*]] = alloca { half, half }, i64 1, align 2
// LLVM: %[[B_ADDR:.*]] = alloca { half, half }, i64 1, align 2
// LLVM: %[[TMP_A:.*]] = load { half, half }, ptr %[[A_ADDR]], align 2
// LLVM: %[[A_REAL:.*]] = extractvalue { half, half } %[[TMP_A]], 0
// LLVM: %[[A_IMAG:.*]] = extractvalue { half, half } %[[TMP_A]], 1
// LLVM: %[[A_REAL_F32:.*]] = fpext half %[[A_REAL]] to float
// LLVM: %[[A_IMAG_F32:.*]] = fpext half %[[A_IMAG]] to float
// LLVM: %[[TMP_A_COMPLEX_F32:.*]] = insertvalue { float, float } {{.*}}, float %[[A_REAL_F32]], 0
// LLVM: %[[A_COMPLEX_F32:.*]] = insertvalue { float, float } %[[TMP_A_COMPLEX_F32]], float %[[A_IMAG_F32]], 1
// LLVM: %[[TMP_A_COMPLEX_F32:.*]] = insertvalue { float, float } {{.*}}, float %[[A_REAL_F32]], 0
// LLVM: %[[A_COMPLEX_F32:.*]] = insertvalue { float, float } %[[TMP_A_COMPLEX_F32]], float %[[A_IMAG_F32]], 1
// LLVM: %[[A_REAL_F16:.*]] = fptrunc float %[[A_REAL_F32]] to half
// LLVM: %[[A_IMAG_F16:.*]] = fptrunc float %[[A_IMAG_F32]] to half
// LLVM: %[[TMP_RESULT_COMPLEX_F16:.*]] = insertvalue { half, half } {{.*}}, half %[[A_REAL_F16]], 0
// LLVM: %[[RESULT_COMPLEX_F16:.*]] = insertvalue { half, half } %[[TMP_RESULT_COMPLEX_F16]], half %[[A_IMAG_F16]], 1
// LLVM: store { half, half } %[[RESULT_COMPLEX_F16]], ptr %[[B_ADDR]], align 2

// OGCG: %[[A_ADDR:.*]] = alloca { half, half }, align 2
// OGCG: %[[B_ADDR:.*]] = alloca { half, half }, align 2
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { half, half }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load half, ptr %a.realp, align 2
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { half, half }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load half, ptr %a.imagp, align 2
// OGCG: %[[A_REAL_F32:.*]] = fpext half %[[A_REAL]] to float
// OGCG: %[[A_IMAG_F32:.*]] = fpext half %[[A_IMAG]] to float
// OGCG: %[[RESULT_REAL:.*]] = fptrunc float %[[A_REAL_F32]] to half
// OGCG: %[[RESULT_IMAG:.*]] = fptrunc float %[[A_IMAG_F32]] to half
// OGCG: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { half, half }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { half, half }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG: store half %[[RESULT_REAL]], ptr %[[B_REAL_PTR]], align 2
// OGCG: store half %[[RESULT_IMAG]], ptr %[[B_IMAG_PTR]], align 2

void foo10() {
  _Float16 _Complex a;
  _Float16 _Complex b = -a;
}

// CIR-BEFORE: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.f16>, !cir.ptr<!cir.complex<!cir.f16>>, ["a"]
// CIR-BEFORE: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!cir.f16>, !cir.ptr<!cir.complex<!cir.f16>>, ["b", init]
// CIR-BEFORE: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.f16>>, !cir.complex<!cir.f16>
// CIR-BEFORE: %[[A_COMPLEX_F32:.*]] = cir.cast(float_complex, %[[TMP_A]] : !cir.complex<!cir.f16>), !cir.complex<!cir.float>
// CIR-BEFORE: %[[RESULT:.*]] = cir.unary(minus, %[[A_COMPLEX_F32]]) : !cir.complex<!cir.float>, !cir.complex<!cir.float>
// CIR-BEFORE: %[[A_COMPLEX_F16:.*]] = cir.cast(float_complex, %[[RESULT]] : !cir.complex<!cir.float>), !cir.complex<!cir.f16>
// CIR-BEFORE: cir.store{{.*}} %[[A_COMPLEX_F16]], %[[B_ADDR]] : !cir.complex<!cir.f16>, !cir.ptr<!cir.complex<!cir.f16>>

// CIR-AFTER: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.f16>, !cir.ptr<!cir.complex<!cir.f16>>, ["a"]
// CIR-AFTER: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!cir.f16>, !cir.ptr<!cir.complex<!cir.f16>>, ["b", init]
// CIR-AFTER: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.f16>>, !cir.complex<!cir.f16>
// CIR-AFTER: %[[A_REAL:.*]] = cir.complex.real %[[TMP_A]] : !cir.complex<!cir.f16> -> !cir.f16
// CIR-AFTER: %[[A_IMAG:.*]] = cir.complex.imag %[[TMP_A]] : !cir.complex<!cir.f16> -> !cir.f16
// CIR-AFTER: %[[A_REAL_F32:.*]] = cir.cast(floating, %[[A_REAL]] : !cir.f16), !cir.float
// CIR-AFTER: %[[A_IMAG_F32:.*]] = cir.cast(floating, %[[A_IMAG]] : !cir.f16), !cir.float
// CIR-AFTER: %[[A_COMPLEX_F32:.*]] = cir.complex.create %[[A_REAL_F32]], %[[A_IMAG_F32]] : !cir.float -> !cir.complex<!cir.float>
// CIR-AFTER: %[[A_REAL_F32:.*]] = cir.complex.real %[[A_COMPLEX_F32]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER: %[[A_IMAG_F32:.*]] = cir.complex.imag %[[A_COMPLEX_F32]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER: %[[RESULT_REAL_F32:.*]] = cir.unary(minus, %[[A_REAL_F32]]) : !cir.float, !cir.float
// CIR-AFTER: %[[RESULT_IMAG_F32:.*]] = cir.unary(minus, %[[A_IMAG_F32]]) : !cir.float, !cir.float
// CIR-AFTER: %[[RESULT_COMPLEX_F32:.*]] = cir.complex.create %[[RESULT_REAL_F32]], %[[RESULT_IMAG_F32]] : !cir.float -> !cir.complex<!cir.float>
// CIR-AFTER: %[[RESULT_REAL_F32:.*]] = cir.complex.real %[[RESULT_COMPLEX_F32]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER: %[[RESULT_IMAG_F32:.*]] = cir.complex.imag %[[RESULT_COMPLEX_F32]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER: %[[RESULT_REAL_F16:.*]] = cir.cast(floating, %[[RESULT_REAL_F32]] : !cir.float), !cir.f16
// CIR-AFTER: %[[RESULT_IMAG_F16:.*]] = cir.cast(floating, %[[RESULT_IMAG_F32]] : !cir.float), !cir.f16
// CIR-AFTER: %[[RESULT_COMPLEX_F16:.*]] = cir.complex.create %[[RESULT_REAL_F16]], %[[RESULT_IMAG_F16]] : !cir.f16 -> !cir.complex<!cir.f16>
// CIR-AFTER: cir.store{{.*}} %[[RESULT_COMPLEX_F16]], %[[B_ADDR]] : !cir.complex<!cir.f16>, !cir.ptr<!cir.complex<!cir.f16>>

// LLVM: %[[A_ADDR:.*]] = alloca { half, half }, i64 1, align 2
// LLVM: %[[B_ADDR:.*]] = alloca { half, half }, i64 1, align 2
// LLVM: %[[TMP_A:.*]] = load { half, half }, ptr %[[A_ADDR]], align 2
// LLVM: %[[A_REAL:.*]] = extractvalue { half, half } %[[TMP_A]], 0
// LLVM: %[[A_IMAG:.*]] = extractvalue { half, half } %[[TMP_A]], 1
// LLVM: %[[A_REAL_F32:.*]] = fpext half %[[A_REAL]] to float
// LLVM: %[[A_IMAG_F32:.*]] = fpext half %[[A_IMAG]] to float
// LLVM: %[[TMP_A_COMPLEX_F32:.*]] = insertvalue { float, float } {{.*}}, float %[[A_REAL_F32]], 0
// LLVM: %[[A_COMPLEX_F32:.*]] = insertvalue { float, float } %[[TMP_A_COMPLEX_F32]], float %[[A_IMAG_F32]], 1
// LLVM: %[[RESULT_REAL_F32:.*]] = fneg float %[[A_REAL_F32]]
// LLVM: %[[RESULT_IMAG_F32:.*]] = fneg float %[[A_IMAG_F32]]
// LLVM: %[[TMP_A_COMPLEX_F32:.*]] = insertvalue { float, float } {{.*}}, float %[[RESULT_REAL_F32]], 0
// LLVM: %[[A_COMPLEX_F32:.*]] = insertvalue { float, float } %[[TMP_A_COMPLEX_F32]], float %[[RESULT_IMAG_F32]], 1
// LLVM: %[[A_REAL_F16:.*]] = fptrunc float %[[RESULT_REAL_F32]] to half
// LLVM: %[[A_IMAG_F16:.*]] = fptrunc float %[[RESULT_IMAG_F32]] to half
// LLVM: %[[TMP_RESULT_COMPLEX_F16:.*]] = insertvalue { half, half } {{.*}}, half %[[A_REAL_F16]], 0
// LLVM: %[[RESULT_COMPLEX_F16:.*]] = insertvalue { half, half } %[[TMP_RESULT_COMPLEX_F16]], half %[[A_IMAG_F16]], 1
// LLVM: store { half, half } %[[RESULT_COMPLEX_F16]], ptr %[[B_ADDR]], align 2

// OGCG: %[[A_ADDR:.*]] = alloca { half, half }, align 2
// OGCG: %[[B_ADDR:.*]] = alloca { half, half }, align 2
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { half, half }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load half, ptr %a.realp, align 2
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { half, half }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load half, ptr %a.imagp, align 2
// OGCG: %[[A_REAL_F32:.*]] = fpext half %[[A_REAL]] to float
// OGCG: %[[A_IMAG_F32:.*]] = fpext half %[[A_IMAG]] to float
// OGCG: %[[RESULT_REAL_F32:.*]] = fneg float %[[A_REAL_F32]]
// OGCG: %[[RESULT_IMAG_F32:.*]] = fneg float %[[A_IMAG_F32]]
// OGCG: %[[RESULT_REAL:.*]] = fptrunc float %[[RESULT_REAL_F32]] to half
// OGCG: %[[RESULT_IMAG:.*]] = fptrunc float %[[RESULT_IMAG_F32]] to half
// OGCG: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { half, half }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { half, half }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG: store half %[[RESULT_REAL]], ptr %[[B_REAL_PTR]], align 2
// OGCG: store half %[[RESULT_IMAG]], ptr %[[B_IMAG_PTR]], align 2
