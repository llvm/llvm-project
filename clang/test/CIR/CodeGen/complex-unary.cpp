// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefixes=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

void foo() {
  int _Complex a;
  int _Complex b = ~a;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["b", init]
// CIR: %[[TMP:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR: %[[COMPLEX_NOT:.*]] = cir.unary(not, %[[TMP]]) : !cir.complex<!s32i>, !cir.complex<!s32i>
// CIR: cir.store{{.*}} %[[COMPLEX_NOT]], %[[B_ADDR]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>

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

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["b", init]
// CIR: %[[TMP:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: %[[COMPLEX_NOT:.*]] = cir.unary(not, %[[TMP]]) : !cir.complex<!cir.float>, !cir.complex<!cir.float>
// CIR: cir.store{{.*}} %[[COMPLEX_NOT]], %[[B_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

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

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["b", init]
// CIR: %[[TMP:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: %[[COMPLEX_INC:.*]] = cir.unary(inc, %[[TMP]]) : !cir.complex<!cir.float>, !cir.complex<!cir.float>
// CIR: cir.store{{.*}} %[[COMPLEX_INC]], %[[A_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>
// CIR: cir.store{{.*}} %[[TMP]], %[[B_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

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

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["b", init]
// CIR: %[[TMP:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: %[[COMPLEX_INC:.*]] = cir.unary(inc, %[[TMP]]) : !cir.complex<!cir.float>, !cir.complex<!cir.float>
// CIR: cir.store{{.*}} %[[COMPLEX_INC]], %[[A_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>
// CIR: cir.store{{.*}} %[[COMPLEX_INC]], %[[B_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

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

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["b", init]
// CIR: %[[TMP:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: %[[COMPLEX_DEC:.*]] = cir.unary(dec, %[[TMP]]) : !cir.complex<!cir.float>, !cir.complex<!cir.float>
// CIR: cir.store{{.*}} %[[COMPLEX_DEC]], %[[A_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>
// CIR: cir.store{{.*}} %[[TMP]], %[[B_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

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

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["b", init]
// CIR: %[[TMP:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: %[[COMPLEX_DEC:.*]] = cir.unary(dec, %[[TMP]]) : !cir.complex<!cir.float>, !cir.complex<!cir.float>
// CIR: cir.store{{.*}} %[[COMPLEX_DEC]], %[[A_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>
// CIR: cir.store{{.*}} %[[COMPLEX_DEC]], %[[B_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

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

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["b", init]
// CIR: %[[TMP:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: %[[COMPLEX_PLUS:.*]] = cir.unary(plus, %[[TMP]]) : !cir.complex<!cir.float>, !cir.complex<!cir.float>
// CIR: cir.store{{.*}} %[[COMPLEX_PLUS]], %[[B_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

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

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["b", init]
// CIR: %[[TMP:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: %[[COMPLEX_MINUS:.*]] = cir.unary(minus, %[[TMP]]) : !cir.complex<!cir.float>, !cir.complex<!cir.float>
// CIR: cir.store{{.*}} %[[COMPLEX_MINUS]], %[[B_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

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
