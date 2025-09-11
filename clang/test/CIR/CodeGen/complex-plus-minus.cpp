// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

void foo() {
  int _Complex a;
  int _Complex b;
  int _Complex c = a + b;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["b"]
// CIR: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR: %[[TMP_B:.*]] = cir.load{{.*}} %[[B_ADDR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR: %[[ADD:.*]] = cir.complex.add %[[TMP_A]], %[[TMP_B]] : !cir.complex<!s32i>

// LLVM: %[[A_ADDR:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[B_ADDR:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[TMP_A:.*]] = load { i32, i32 }, ptr %[[A_ADDR]], align 4
// LLVM: %[[TMP_B:.*]] = load { i32, i32 }, ptr %[[B_ADDR]], align 4
// LLVM: %[[A_REAL:.*]] = extractvalue { i32, i32 } %[[TMP_A]], 0
// LLVM: %[[A_IMAG:.*]] = extractvalue { i32, i32 } %[[TMP_A]], 1
// LLVM: %[[B_REAL:.*]] = extractvalue { i32, i32 } %[[TMP_B]], 0
// LLVM: %[[B_IMAG:.*]] = extractvalue { i32, i32 } %[[TMP_B]], 1
// LLVM: %[[ADD_REAL:.*]] = add i32 %[[A_REAL]], %[[B_REAL]]
// LLVM: %[[ADD_IMAG:.*]] = add i32 %[[A_IMAG]], %[[B_IMAG]]
// LLVM: %[[RESULT:.*]] = insertvalue { i32, i32 } poison, i32 %[[ADD_REAL]], 0
// LLVM: %[[RESULT_2:.*]] = insertvalue { i32, i32 } %[[RESULT]], i32 %[[ADD_IMAG]], 1

// OGCG: %[[A_ADDR:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[B_ADDR:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[RESULT:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load i32, ptr %[[A_REAL_PTR]], align 4
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load i32, ptr %[[A_IMAG_PTR]], align 4
// OGCG: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG: %[[B_REAL:.*]] = load i32, ptr %[[B_REAL_PTR]], align 4
// OGCG: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG: %[[B_IMAG:.*]] = load i32, ptr %[[B_IMAG_PTR]], align 4
// OGCG: %[[ADD_REAL:.*]] = add i32 %[[A_REAL]], %[[B_REAL]]
// OGCG: %[[ADD_IMAG:.*]] = add i32 %[[A_IMAG]], %[[B_IMAG]]
// OGCG: %[[RESULT_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[RESULT]], i32 0, i32 0
// OGCG: %[[RESULT_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[RESULT]], i32 0, i32 1
// OGCG: store i32 %[[ADD_REAL]], ptr %[[RESULT_REAL_PTR]], align 4
// OGCG: store i32 %[[ADD_IMAG]], ptr %[[RESULT_IMAG_PTR]], align 4

void foo2() {
  float _Complex a;
  float _Complex b;
  float _Complex c = a + b;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["b"]
// CIR: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: %[[TMP_B:.*]] = cir.load{{.*}} %[[B_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: %[[ADD:.*]] = cir.complex.add %[[TMP_A]], %[[TMP_B]] : !cir.complex<!cir.float>

// LLVM: %[[A_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[B_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[TMP_A:.*]] = load { float, float }, ptr %[[A_ADDR]], align 4
// LLVM: %[[TMP_B:.*]] = load { float, float }, ptr %[[B_ADDR]], align 4
// LLVM: %[[A_REAL:.*]] = extractvalue { float, float } %[[TMP_A]], 0
// LLVM: %[[A_IMAG:.*]] = extractvalue { float, float } %[[TMP_A]], 1
// LLVM: %[[B_REAL:.*]] = extractvalue { float, float } %[[TMP_B]], 0
// LLVM: %[[B_IMAG:.*]] = extractvalue { float, float } %[[TMP_B]], 1
// LLVM: %[[ADD_REAL:.*]] = fadd float %[[A_REAL]], %[[B_REAL]]
// LLVM: %[[ADD_IMAG:.*]] = fadd float %[[A_IMAG]], %[[B_IMAG]]
// LLVM: %[[RESULT:.*]] = insertvalue { float, float } poison, float %[[ADD_REAL]], 0
// LLVM: %[[RESULT_2:.*]] = insertvalue { float, float } %[[RESULT]], float %[[ADD_IMAG]], 1

// OGCG: %[[A_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[B_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[RESULT:.*]] = alloca { float, float }, align 4
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load float, ptr %[[A_REAL_PTR]], align 4
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load float, ptr %[[A_IMAG_PTR]], align 4
// OGCG: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG: %[[B_REAL:.*]] = load float, ptr %[[B_REAL_PTR]], align 4
// OGCG: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG: %[[B_IMAG:.*]] = load float, ptr %[[B_IMAG_PTR]], align 4
// OGCG: %[[ADD_REAL:.*]] = fadd float %[[A_REAL]], %[[B_REAL]]
// OGCG: %[[ADD_IMAG:.*]] = fadd float %[[A_IMAG]], %[[B_IMAG]]
// OGCG: %[[RESULT_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[RESULT]], i32 0, i32 0
// OGCG: %[[RESULT_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[RESULT]], i32 0, i32 1
// OGCG: store float %[[ADD_REAL]], ptr %[[RESULT_REAL_PTR]], align 4
// OGCG: store float %[[ADD_IMAG]], ptr %[[RESULT_IMAG_PTR]], align 4

void foo3() {
  float _Complex a;
  float _Complex b;
  float _Complex c;
  float _Complex d = (a + b) + c;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["b"]
// CIR: %[[C_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["c"]
// CIR: %[[RESULT:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["d", init]
// CIR: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: %[[TMP_B:.*]] = cir.load{{.*}} %[[B_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: %[[ADD_A_B:.*]] = cir.complex.add %[[TMP_A]], %[[TMP_B]] : !cir.complex<!cir.float>
// CIR: %[[TMP_C:.*]] = cir.load{{.*}} %[[C_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: %[[ADD_A_B_C:.*]] = cir.complex.add %[[ADD_A_B]], %[[TMP_C]] : !cir.complex<!cir.float>
// CIR: cir.store{{.*}} %[[ADD_A_B_C]], %[[RESULT]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// LLVM: %[[A_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[B_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[C_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[RESULT:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[TMP_A:.*]] = load { float, float }, ptr %[[A_ADDR]], align 4
// LLVM: %[[TMP_B:.*]] = load { float, float }, ptr %[[B_ADDR]], align 4
// LLVM: %[[A_REAL:.*]] = extractvalue { float, float } %[[TMP_A]], 0
// LLVM: %[[A_IMAG:.*]] = extractvalue { float, float } %[[TMP_A]], 1
// LLVM: %[[B_REAL:.*]] = extractvalue { float, float } %[[TMP_B]], 0
// LLVM: %[[B_IMAG:.*]] = extractvalue { float, float } %[[TMP_B]], 1
// LLVM: %[[ADD_REAL_A_B:.*]] = fadd float %[[A_REAL]], %[[B_REAL]]
// LLVM: %[[ADD_IMAG_A_B:.*]] = fadd float %[[A_IMAG]], %[[B_IMAG]]
// LLVM: %[[A_B:.*]] = insertvalue { float, float } poison, float %[[ADD_REAL_A_B]], 0
// LLVM: %[[TMP_A_B:.*]] = insertvalue { float, float } %[[A_B]], float %[[ADD_IMAG_A_B]], 1
// LLVM: %[[TMP_C:.*]] = load { float, float }, ptr %[[C_ADDR]], align 4
// LLVM: %[[A_B_REAL:.*]] = extractvalue { float, float } %[[TMP_A_B]], 0
// LLVM: %[[A_B_IMAG:.*]] = extractvalue { float, float } %[[TMP_A_B]], 1
// LLVM: %[[C_REAL:.*]] = extractvalue { float, float } %[[TMP_C]], 0
// LLVM: %[[C_IMAG:.*]] = extractvalue { float, float } %[[TMP_C]], 1
// LLVM: %[[ADD_REAL_A_B_C:.*]] = fadd float %[[A_B_REAL]], %[[C_REAL]]
// LLVM: %[[ADD_IMAG_A_B_C:.*]] = fadd float %[[A_B_IMAG]], %[[C_IMAG]]
// LLVM: %[[A_B_C:.*]] = insertvalue { float, float } poison, float %[[ADD_REAL_A_B_C]], 0
// LLVM: %[[TMP_A_B_C:.*]] = insertvalue { float, float } %[[A_B_C]], float %[[ADD_IMAG_A_B_C]], 1
// LLVM: store { float, float } %[[TMP_A_B_C]], ptr %[[RESULT]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[B_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[C_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[RESULT:.*]] = alloca { float, float }, align 4
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load float, ptr %[[A_REAL_PTR]], align 4
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load float, ptr %[[A_IMAG_PTR]], align 4
// OGCG: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG: %[[B_REAL:.*]] = load float, ptr %[[B_REAL_PTR]], align 4
// OGCG: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG: %[[B_IMAG:.*]] = load float, ptr %[[B_IMAG_PTR]], align 4
// OGCG: %[[ADD_REAL_A_B:.*]] = fadd float %[[A_REAL]], %[[B_REAL]]
// OGCG: %[[ADD_IMAG_A_B:.*]] = fadd float %[[A_IMAG]], %[[B_IMAG]]
// OGCG: %[[C_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[C_ADDR]], i32 0, i32 0
// OGCG: %[[C_REAL:.*]] = load float, ptr %[[C_REAL_PTR]], align 4
// OGCG: %[[C_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[C_ADDR]], i32 0, i32 1
// OGCG: %[[C_IMAG:.*]] = load float, ptr %[[C_IMAG_PTR]], align 4
// OGCG: %[[ADD_REAL_A_B_C:.*]] = fadd float %[[ADD_REAL_A_B]], %[[C_REAL]]
// OGCG: %[[ADD_IMAG_A_B_C:.*]] = fadd float %[[ADD_IMAG_A_B]], %[[C_IMAG]]
// OGCG: %[[RESULT_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[RESULT]], i32 0, i32 0
// OGCG: %[[RESULT_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[RESULT]], i32 0, i32 1
// OGCG: store float %[[ADD_REAL_A_B_C]], ptr %[[RESULT_REAL_PTR]], align 4
// OGCG: store float %[[ADD_IMAG_A_B_C]], ptr %[[RESULT_IMAG_PTR]], align 4

void foo4() {
  int _Complex a;
  int _Complex b;
  int _Complex c = a - b;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["b"]
// CIR: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR: %[[TMP_B:.*]] = cir.load{{.*}} %[[B_ADDR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR: %[[SUB:.*]] = cir.complex.sub %[[TMP_A]], %[[TMP_B]] : !cir.complex<!s32i>

// LLVM: %[[A_ADDR:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[B_ADDR:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[C_ADDR:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[TMP_A:.*]] = load { i32, i32 }, ptr %[[A_ADDR]], align 4
// LLVM: %[[TMP_B:.*]] = load { i32, i32 }, ptr %[[B_ADDR]], align 4
// LLVM: %[[A_REAL:.*]] = extractvalue { i32, i32 } %[[TMP_A]], 0
// LLVM: %[[A_IMAG:.*]] = extractvalue { i32, i32 } %[[TMP_A]], 1
// LLVM: %[[B_REAL:.*]] = extractvalue { i32, i32 } %[[TMP_B]], 0
// LLVM: %[[B_IMAG:.*]] = extractvalue { i32, i32 } %[[TMP_B]], 1
// LLVM: %[[SUB_REAL:.*]] = sub i32 %[[A_REAL]], %[[B_REAL]]
// LLVM: %[[SUB_IMAG:.*]] = sub i32 %[[A_IMAG]], %[[B_IMAG]]
// LLVM: %[[RESULT:.*]] = insertvalue { i32, i32 } poison, i32 %[[SUB_REAL]], 0
// LLVM: %[[RESULT_2:.*]] = insertvalue { i32, i32 } %[[RESULT]], i32 %[[SUB_IMAG]], 1
// LLVM: store { i32, i32 } %[[RESULT_2]], ptr %[[C_ADDR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[B_ADDR:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[RESULT:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load i32, ptr %[[A_REAL_PTR]], align 4
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load i32, ptr %[[A_IMAG_PTR]], align 4
// OGCG: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG: %[[B_REAL:.*]] = load i32, ptr %[[B_REAL_PTR]], align 4
// OGCG: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG: %[[B_IMAG:.*]] = load i32, ptr %[[B_IMAG_PTR]], align 4
// OGCG: %[[SUB_REAL:.*]] = sub i32 %[[A_REAL]], %[[B_REAL]]
// OGCG: %[[SUB_IMAG:.*]] = sub i32 %[[A_IMAG]], %[[B_IMAG]]
// OGCG: %[[RESULT_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[RESULT]], i32 0, i32 0
// OGCG: %[[RESULT_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[RESULT]], i32 0, i32 1
// OGCG: store i32 %[[SUB_REAL]], ptr %[[RESULT_REAL_PTR]], align 4
// OGCG: store i32 %[[SUB_IMAG]], ptr %[[RESULT_IMAG_PTR]], align 4

void foo5() {
  float _Complex a;
  float _Complex b;
  float _Complex c = a - b;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["b"]
// CIR: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: %[[TMP_B:.*]] = cir.load{{.*}} %[[B_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: %[[SUB:.*]] = cir.complex.sub %[[TMP_A]], %[[TMP_B]] : !cir.complex<!cir.float>

// LLVM: %[[A_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[B_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[TMP_A:.*]] = load { float, float }, ptr %[[A_ADDR]], align 4
// LLVM: %[[TMP_B:.*]] = load { float, float }, ptr %[[B_ADDR]], align 4
// LLVM: %[[A_REAL:.*]] = extractvalue { float, float } %[[TMP_A]], 0
// LLVM: %[[A_IMAG:.*]] = extractvalue { float, float } %[[TMP_A]], 1
// LLVM: %[[B_REAL:.*]] = extractvalue { float, float } %[[TMP_B]], 0
// LLVM: %[[B_IMAG:.*]] = extractvalue { float, float } %[[TMP_B]], 1
// LLVM: %[[SUB_REAL:.*]] = fsub float %[[A_REAL]], %[[B_REAL]]
// LLVM: %[[SUB_IMAG:.*]] = fsub float %[[A_IMAG]], %[[B_IMAG]]
// LLVM: %[[RESULT:.*]] = insertvalue { float, float } poison, float %[[SUB_REAL]], 0
// LLVM: %[[RESULT_2:.*]] = insertvalue { float, float } %[[RESULT]], float %[[SUB_IMAG]], 1

// OGCG: %[[A_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[B_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[RESULT:.*]] = alloca { float, float }, align 4
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load float, ptr %[[A_REAL_PTR]], align 4
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load float, ptr %[[A_IMAG_PTR]], align 4
// OGCG: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG: %[[B_REAL:.*]] = load float, ptr %[[B_REAL_PTR]], align 4
// OGCG: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG: %[[B_IMAG:.*]] = load float, ptr %[[B_IMAG_PTR]], align 4
// OGCG: %[[SUB_REAL:.*]] = fsub float %[[A_REAL]], %[[B_REAL]]
// OGCG: %[[SUB_IMAG:.*]] = fsub float %[[A_IMAG]], %[[B_IMAG]]
// OGCG: %[[RESULT_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[RESULT]], i32 0, i32 0
// OGCG: %[[RESULT_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[RESULT]], i32 0, i32 1
// OGCG: store float %[[SUB_REAL]], ptr %[[RESULT_REAL_PTR]], align 4
// OGCG: store float %[[SUB_IMAG]], ptr %[[RESULT_IMAG_PTR]], align 4

void foo6() {
  float _Complex a;
  float _Complex b;
  float _Complex c;
  float _Complex d = (a - b) - c;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["b"]
// CIR: %[[C_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["c"]
// CIR: %[[RESULT:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["d", init]
// CIR: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: %[[TMP_B:.*]] = cir.load{{.*}} %[[B_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: %[[SUB_A_B:.*]] = cir.complex.sub %[[TMP_A]], %[[TMP_B]] : !cir.complex<!cir.float>
// CIR: %[[TMP_C:.*]] = cir.load{{.*}} %[[C_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: %[[SUB_A_B_C:.*]] = cir.complex.sub %[[SUB_A_B]], %[[TMP_C]] : !cir.complex<!cir.float>
// CIR: cir.store{{.*}} %[[SUB_A_B_C]], %[[RESULT]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// LLVM: %[[A_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[B_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[C_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[RESULT:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[TMP_A:.*]] = load { float, float }, ptr %[[A_ADDR]], align 4
// LLVM: %[[TMP_B:.*]] = load { float, float }, ptr %[[B_ADDR]], align 4
// LLVM: %[[A_REAL:.*]] = extractvalue { float, float } %[[TMP_A]], 0
// LLVM: %[[A_IMAG:.*]] = extractvalue { float, float } %[[TMP_A]], 1
// LLVM: %[[B_REAL:.*]] = extractvalue { float, float } %[[TMP_B]], 0
// LLVM: %[[B_IMAG:.*]] = extractvalue { float, float } %[[TMP_B]], 1
// LLVM: %[[SUB_REAL_A_B:.*]] = fsub float %[[A_REAL]], %[[B_REAL]]
// LLVM: %[[SUB_IMAG_A_B:.*]] = fsub float %[[A_IMAG]], %[[B_IMAG]]
// LLVM: %[[A_B:.*]] = insertvalue { float, float } poison, float %[[SUB_REAL_A_B]], 0
// LLVM: %[[TMP_A_B:.*]] = insertvalue { float, float } %[[A_B]], float %[[SUB_IMAG_A_B]], 1
// LLVM: %[[TMP_C:.*]] = load { float, float }, ptr %[[C_ADDR]], align 4
// LLVM: %[[A_B_REAL:.*]] = extractvalue { float, float } %[[TMP_A_B]], 0
// LLVM: %[[A_B_IMAG:.*]] = extractvalue { float, float } %[[TMP_A_B]], 1
// LLVM: %[[C_REAL:.*]] = extractvalue { float, float } %[[TMP_C]], 0
// LLVM: %[[C_IMAG:.*]] = extractvalue { float, float } %[[TMP_C]], 1
// LLVM: %[[SUB_REAL_A_B_C:.*]] = fsub float %[[A_B_REAL]], %[[C_REAL]]
// LLVM: %[[SUB_IMAG_A_B_C:.*]] = fsub float %[[A_B_IMAG]], %[[C_IMAG]]
// LLVM: %[[A_B_C:.*]] = insertvalue { float, float } poison, float %[[SUB_REAL_A_B_C]], 0
// LLVM: %[[TMP_A_B_C:.*]] = insertvalue { float, float } %[[A_B_C]], float %[[SUB_IMAG_A_B_C]], 1
// LLVM: store { float, float } %[[TMP_A_B_C]], ptr %[[RESULT]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[B_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[C_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[RESULT:.*]] = alloca { float, float }, align 4
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load float, ptr %[[A_REAL_PTR]], align 4
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load float, ptr %[[A_IMAG_PTR]], align 4
// OGCG: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG: %[[B_REAL:.*]] = load float, ptr %[[B_REAL_PTR]], align 4
// OGCG: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG: %[[B_IMAG:.*]] = load float, ptr %[[B_IMAG_PTR]], align 4
// OGCG: %[[SUB_REAL_A_B:.*]] = fsub float %[[A_REAL]], %[[B_REAL]]
// OGCG: %[[SUB_IMAG_A_B:.*]] = fsub float %[[A_IMAG]], %[[B_IMAG]]
// OGCG: %[[C_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[C_ADDR]], i32 0, i32 0
// OGCG: %[[C_REAL:.*]] = load float, ptr %[[C_REAL_PTR]], align 4
// OGCG: %[[C_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[C_ADDR]], i32 0, i32 1
// OGCG: %[[C_IMAG:.*]] = load float, ptr %[[C_IMAG_PTR]], align 4
// OGCG: %[[SUB_REAL_A_B_C:.*]] = fsub float %[[SUB_REAL_A_B]], %[[C_REAL]]
// OGCG: %[[SUB_IMAG_A_B_C:.*]] = fsub float %[[SUB_IMAG_A_B]], %[[C_IMAG]]
// OGCG: %[[RESULT_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[RESULT]], i32 0, i32 0
// OGCG: %[[RESULT_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[RESULT]], i32 0, i32 1
// OGCG: store float %[[SUB_REAL_A_B_C]], ptr %[[RESULT_REAL_PTR]], align 4
// OGCG: store float %[[SUB_IMAG_A_B_C]], ptr %[[RESULT_IMAG_PTR]], align 4

