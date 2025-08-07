// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefixes=CIR,CXX_CIR
// RUN: %clang_cc1 -x c -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefixes=LLVM,CXX_LLVM
// RUN: %clang_cc1 -x c -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefixes=OGCG,CXX_OGCG
// RUN: %clang_cc1 -x c -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

void foo() {
  float _Complex a;
  float _Complex b;
  b += a;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["b"]
// CIR: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: %[[TMP_B:.*]] = cir.load{{.*}} %[[B_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: %[[RESULT:.*]] = cir.complex.add %[[TMP_B]], %[[TMP_A]] : !cir.complex<!cir.float>
// CIR: cir.store{{.*}} %[[RESULT]], %[[B_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// LLVM: %[[A_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[B_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[TMP_A:.*]] = load { float, float }, ptr %[[A_ADDR]], align 4
// LLVM: %[[TMP_B:.*]] = load { float, float }, ptr %[[B_ADDR]], align 4
// LLVM: %[[B_REAL:.*]] = extractvalue { float, float } %[[TMP_B]], 0
// LLVM: %[[B_IMAG:.*]] = extractvalue { float, float } %[[TMP_B]], 1
// LLVM: %[[A_REAL:.*]] = extractvalue { float, float } %[[TMP_A]], 0
// LLVM: %[[A_IMAG:.*]] = extractvalue { float, float } %[[TMP_A]], 1
// LLVM: %[[ADD_REAL_A_B:.*]] = fadd float %[[B_REAL]], %[[A_REAL]]
// LLVM: %[[ADD_IMAG_A_B:.*]] = fadd float %[[B_IMAG]], %[[A_IMAG]]
// LLVM: %[[ADD_A_B:.*]] = insertvalue { float, float } poison, float %[[ADD_REAL_A_B]], 0
// LLVM: %[[RESULT:.*]] = insertvalue { float, float } %[[ADD_A_B]], float %[[ADD_IMAG_A_B]], 1
// LLVM: store { float, float } %[[RESULT]], ptr %[[B_ADDR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[B_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load float, ptr %[[A_REAL_PTR]], align 4
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load float, ptr %[[A_IMAG_PTR]], align 4
// OGCG: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG: %[[B_REAL:.*]] = load float, ptr %[[B_REAL_PTR]], align 4
// OGCG: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG: %[[B_IMAG:.*]] = load float, ptr %[[B_IMAG_PTR]], align 4
// OGCG: %[[ADD_REAL:.*]] = fadd float %[[B_REAL]], %[[A_REAL]]
// OGCG: %[[ADD_IMAG:.*]] = fadd float %[[B_IMAG]], %[[A_IMAG]]
// OGCG: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG: store float %[[ADD_REAL]], ptr %[[B_REAL_PTR]], align 4
// OGCG: store float %[[ADD_IMAG]], ptr %[[B_IMAG_PTR]], align 4

void foo1() {
  float _Complex a;
  float _Complex b;
  b -= a;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["b"]
// CIR: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: %[[TMP_B:.*]] = cir.load{{.*}} %[[B_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: %[[RESULT:.*]] = cir.complex.sub %[[TMP_B]], %[[TMP_A]] : !cir.complex<!cir.float>
// CIR: cir.store{{.*}} %[[RESULT]], %[[B_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// LLVM: %[[A_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[B_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[TMP_A:.*]] = load { float, float }, ptr %[[A_ADDR]], align 4
// LLVM: %[[TMP_B:.*]] = load { float, float }, ptr %[[B_ADDR]], align 4
// LLVM: %[[B_REAL:.*]] = extractvalue { float, float } %[[TMP_B]], 0
// LLVM: %[[B_IMAG:.*]] = extractvalue { float, float } %[[TMP_B]], 1
// LLVM: %[[A_REAL:.*]] = extractvalue { float, float } %[[TMP_A]], 0
// LLVM: %[[A_IMAG:.*]] = extractvalue { float, float } %[[TMP_A]], 1
// LLVM: %[[SUB_REAL_A_B:.*]] = fsub float %[[B_REAL]], %[[A_REAL]]
// LLVM: %[[SUB_IMAG_A_B:.*]] = fsub float %[[B_IMAG]], %[[A_IMAG]]
// LLVM: %[[SUB_A_B:.*]] = insertvalue { float, float } poison, float %[[SUB_REAL_A_B]], 0
// LLVM: %[[RESULT:.*]] = insertvalue { float, float } %[[SUB_A_B]], float %[[SUB_IMAG_A_B]], 1
// LLVM: store { float, float } %[[RESULT]], ptr %[[B_ADDR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[B_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load float, ptr %[[A_REAL_PTR]], align 4
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load float, ptr %[[A_IMAG_PTR]], align 4
// OGCG: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG: %[[B_REAL:.*]] = load float, ptr %[[B_REAL_PTR]], align 4
// OGCG: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG: %[[B_IMAG:.*]] = load float, ptr %[[B_IMAG_PTR]], align 4
// OGCG: %[[SUB_REAL:.*]] = fsub float %[[B_REAL]], %[[A_REAL]]
// OGCG: %[[SUB_IMAG:.*]] = fsub float %[[B_IMAG]], %[[A_IMAG]]
// OGCG: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG: store float %[[SUB_REAL]], ptr %[[B_REAL_PTR]], align 4
// OGCG: store float %[[SUB_IMAG]], ptr %[[B_IMAG_PTR]], align 4

void foo2() {
  int _Complex a;
  int _Complex b;
  b += a;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["b"]
// CIR: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR: %[[TMP_B:.*]] = cir.load{{.*}} %[[B_ADDR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR: %[[RESULT:.*]] = cir.complex.add %[[TMP_B]], %[[TMP_A]] : !cir.complex<!s32i>
// CIR: cir.store{{.*}} %[[RESULT]], %[[B_ADDR]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>

// LLVM: %[[A_ADDR:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[B_ADDR:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[TMP_A:.*]] = load { i32, i32 }, ptr %[[A_ADDR]], align 4
// LLVM: %[[TMP_B:.*]] = load { i32, i32 }, ptr %[[B_ADDR]], align 4
// LLVM: %[[B_REAL:.*]] = extractvalue { i32, i32 } %[[TMP_B]], 0
// LLVM: %[[B_IMAG:.*]] = extractvalue { i32, i32 } %[[TMP_B]], 1
// LLVM: %[[A_REAL:.*]] = extractvalue { i32, i32 } %[[TMP_A]], 0
// LLVM: %[[A_IMAG:.*]] = extractvalue { i32, i32 } %[[TMP_A]], 1
// LLVM: %[[ADD_REAL_A_B:.*]] = add i32 %[[B_REAL]], %[[A_REAL]]
// LLVM: %[[ADD_IMAG_A_B:.*]] = add i32 %[[B_IMAG]], %[[A_IMAG]]
// LLVM: %[[ADD_A_B:.*]] = insertvalue { i32, i32 } poison, i32 %[[ADD_REAL_A_B]], 0
// LLVM: %[[RESULT:.*]] = insertvalue { i32, i32 } %[[ADD_A_B]], i32 %[[ADD_IMAG_A_B]], 1
// LLVM: store { i32, i32 } %[[RESULT]], ptr %[[B_ADDR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[B_ADDR:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load i32, ptr %[[A_REAL_PTR]], align 4
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load i32, ptr %[[A_IMAG_PTR]], align 4
// OGCG: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG: %[[B_REAL:.*]] = load i32, ptr %[[B_REAL_PTR]], align 4
// OGCG: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG: %[[B_IMAG:.*]] = load i32, ptr %[[B_IMAG_PTR]], align 4
// OGCG: %[[ADD_REAL:.*]] = add i32 %[[B_REAL]], %[[A_REAL]]
// OGCG: %[[ADD_IMAG:.*]] = add i32 %[[B_IMAG]], %[[A_IMAG]]
// OGCG: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG: store i32 %[[ADD_REAL]], ptr %[[B_REAL_PTR]], align 4
// OGCG: store i32 %[[ADD_IMAG]], ptr %[[B_IMAG_PTR]], align 4

void foo3() {
  _Float16 _Complex a;
  _Float16 _Complex b;
  b += a;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.f16>, !cir.ptr<!cir.complex<!cir.f16>>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!cir.f16>, !cir.ptr<!cir.complex<!cir.f16>>, ["b"]
// CIR: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.f16>>, !cir.complex<!cir.f16>
// CIR: %[[A_REAL:.*]] = cir.complex.real %[[TMP_A]] : !cir.complex<!cir.f16> -> !cir.f16
// CIR: %[[A_IMAG:.*]] = cir.complex.imag %[[TMP_A]] : !cir.complex<!cir.f16> -> !cir.f16
// CIR: %[[A_REAL_F32:.*]] = cir.cast(floating, %[[A_REAL]] : !cir.f16), !cir.float
// CIR: %[[A_IMAG_F32:.*]] = cir.cast(floating, %[[A_IMAG]] : !cir.f16), !cir.float
// CIR: %[[A_COMPLEX_F32:.*]] = cir.complex.create %[[A_REAL_F32]], %[[A_IMAG_F32]] : !cir.float -> !cir.complex<!cir.float>
// CIR: %[[TMP_B:.*]] = cir.load{{.*}} %[[B_ADDR]] : !cir.ptr<!cir.complex<!cir.f16>>, !cir.complex<!cir.f16>
// CIR: %[[B_REAL:.*]] = cir.complex.real %[[TMP_B]] : !cir.complex<!cir.f16> -> !cir.f16
// CIR: %[[B_IMAG:.*]] = cir.complex.imag %[[TMP_B]] : !cir.complex<!cir.f16> -> !cir.f16
// CIR: %[[B_REAL_F32:.*]] = cir.cast(floating, %[[B_REAL]] : !cir.f16), !cir.float
// CIR: %[[B_IMAG_F32:.*]] = cir.cast(floating, %[[B_IMAG]] : !cir.f16), !cir.float
// CIR: %[[B_COMPLEX_F32:.*]] = cir.complex.create %[[B_REAL_F32]], %[[B_IMAG_F32]] : !cir.float -> !cir.complex<!cir.float>
// CIR: %[[ADD_A_B:.*]] = cir.complex.add %[[B_COMPLEX_F32]], %[[A_COMPLEX_F32]] : !cir.complex<!cir.float>
// CIR: %[[ADD_REAL:.*]] = cir.complex.real %[[ADD_A_B]] : !cir.complex<!cir.float> -> !cir.float
// CIR: %[[ADD_IMAG:.*]] = cir.complex.imag %[[ADD_A_B]] : !cir.complex<!cir.float> -> !cir.float
// CIR: %[[ADD_REAL_F16:.*]] = cir.cast(floating, %[[ADD_REAL]] : !cir.float), !cir.f16
// CIR: %[[ADD_IMAG_F16:.*]] = cir.cast(floating, %[[ADD_IMAG]] : !cir.float), !cir.f16
// CIR: %[[RESULT:.*]] = cir.complex.create %[[ADD_REAL_F16]], %[[ADD_IMAG_F16]] : !cir.f16 -> !cir.complex<!cir.f16>
// CIR: cir.store{{.*}} %[[RESULT]], %[[B_ADDR]] : !cir.complex<!cir.f16>, !cir.ptr<!cir.complex<!cir.f16>>

// LLVM: %[[A_ADDR:.*]] = alloca { half, half }, i64 1, align 2
// LLVM: %[[B_ADDR:.*]] = alloca { half, half }, i64 1, align 2
// LLVM: %[[TMP_A:.*]] = load { half, half }, ptr %[[A_ADDR]], align 2
// LLVM: %[[A_REAL:.*]] = extractvalue { half, half } %[[TMP_A]], 0
// LLVM: %[[A_IMAG:.*]] = extractvalue { half, half } %[[TMP_A]], 1
// LLVM: %[[A_REAL_F32:.*]] = fpext half %[[A_REAL]] to float
// LLVM: %[[A_IMAG_F32:.*]] = fpext half %[[A_IMAG]] to float
// LLVM: %[[TMP_A_COMPLEX_F32:.*]] = insertvalue { float, float } {{.*}}, float %[[A_REAL_F32]], 0
// LLVM: %[[A_COMPLEX_F32:.*]] = insertvalue { float, float } %8, float %[[A_IMAG_F32]], 1
// LLVM: %[[TMP_B:.*]] = load { half, half }, ptr %[[B_ADDR]], align 2
// LLVM: %[[B_REAL:.*]] = extractvalue { half, half } %[[TMP_B]], 0
// LLVM: %[[B_IMAG:.*]] = extractvalue { half, half } %[[TMP_B]], 1
// LLVM: %[[B_REAL_F32:.*]] = fpext half %[[B_REAL]] to float
// LLVM: %[[B_IMAG_F32:.*]] = fpext half %[[B_IMAG]] to float
// LLVM: %[[TMP_B_COMPLEX_F32:.*]] = insertvalue { float, float } {{.*}}, float %[[B_REAL_F32]], 0
// LLVM: %[[B_COMPLEX_F32:.*]] = insertvalue { float, float } %[[TMP_B_COMPLEX_F32]], float %[[B_IMAG_F32]], 1
// LLVM: %[[B_REAL:.*]] = extractvalue { float, float } %[[B_COMPLEX_F32]], 0
// LLVM: %[[B_IMAG:.*]] = extractvalue { float, float } %[[B_COMPLEX_F32]], 1
// LLVM: %[[A_REAL:.*]] = extractvalue { float, float } %[[A_COMPLEX_F32]], 0
// LLVM: %[[A_IMAG:.*]] = extractvalue { float, float } %[[A_COMPLEX_F32]], 1
// LLVM: %[[ADD_REAL:.*]] = fadd float %[[B_REAL]], %[[A_REAL]]
// LLVM: %[[ADD_IMAG:.*]] = fadd float %[[B_IMAG]], %[[A_IMAG]]
// LLVM: %[[TMP_RESULT:.*]] = insertvalue { float, float } poison, float %[[ADD_REAL]], 0
// LLVM: %[[RESULT:.*]] = insertvalue { float, float } %[[TMP_RESULT]], float %[[ADD_IMAG]], 1
// LLVM: %[[RESULT_REAL:.*]] = extractvalue { float, float } %[[RESULT]], 0
// LLVM: %[[RESULT_IMAG:.*]] = extractvalue { float, float } %[[RESULT]], 1
// LLVM: %[[RESULT_REAL_F16:.*]] = fptrunc float %[[RESULT_REAL]] to half
// LLVM: %[[RESULT_IMAG_F26:.*]] = fptrunc float %[[RESULT_IMAG]] to half
// LLVM: %[[TMP_RESULT_F16:.*]] = insertvalue { half, half } undef, half %[[RESULT_REAL_F16]], 0
// LLVM: %[[RESULT_F16:.*]] = insertvalue { half, half } %29, half %[[RESULT_IMAG_F26]], 1
// LLVM: store { half, half } %[[RESULT_F16]], ptr %[[B_ADDR]], align 2

// OGCG: %[[A_ADDR:.*]] = alloca { half, half }, align 2
// OGCG: %[[B_ADDR:.*]] = alloca { half, half }, align 2
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { half, half }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load half, ptr %[[A_REAL_PTR]], align 2
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { half, half }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load half, ptr %[[A_IMAG_PTR]], align 2
// OGCG: %[[A_REAL_F32:.*]] = fpext half %[[A_REAL]] to float
// OGCG: %[[A_IMAG_F32:.*]] = fpext half %[[A_IMAG]] to float
// OGCG: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { half, half }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG: %[[B_REAL:.*]] = load half, ptr %[[B_REAL_PTR]], align 2
// OGCG: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { half, half }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG: %[[B_IMAG:.*]] = load half, ptr %[[B_IMAG_PTR]], align 2
// OGCG: %[[B_REAL_F32:.*]] = fpext half %[[B_REAL]] to float
// OGCG: %[[B_IMAG_F32:.*]] = fpext half %[[B_IMAG]] to float
// OGCG: %[[ADD_REAL:.*]] = fadd float %[[B_REAL_F32]], %[[A_REAL_F32]]
// OGCG: %[[ADD_IMAG:.*]] = fadd float %[[B_IMAG_F32]], %[[A_IMAG_F32]]
// OGCG: %[[ADD_REAL_F16:.*]] = fptrunc float %[[ADD_REAL]] to half
// OGCG: %[[ADD_IMAG_F16:.*]] = fptrunc float %[[ADD_IMAG]] to half
// OGCG: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { half, half }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { half, half }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG: store half %[[ADD_REAL_F16]], ptr %[[B_REAL_PTR]], align 2
// OGCG: store half %[[ADD_IMAG_F16]], ptr %[[B_IMAG_PTR]], align 2

#ifdef __cplusplus
void foo4() {
  volatile _Complex int a;
  volatile _Complex int b;
  int _Complex c = b += a;
}
#endif

// CXX_CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["a"]
// CXX_CIR: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["b"]
// CXX_CIR: %[[C_ADDR:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["c", init]
// CXX_CIR: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CXX_CIR: %[[TMP_B:.*]] = cir.load{{.*}} %[[B_ADDR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CXX_CIR: %[[RESULT:.*]] = cir.complex.add %[[TMP_B]], %[[TMP_A]] : !cir.complex<!s32i>
// CXX_CIR: cir.store{{.*}} %[[RESULT]], %[[B_ADDR]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>
// CXX_CIR: %[[TMP_B:.*]] = cir.load{{.*}} %[[B_ADDR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CXX_CIR: cir.store{{.*}} %[[TMP_B]], %[[C_ADDR]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>

// CXX_LLVM: %[[A_ADDR:.*]] = alloca { i32, i32 }, i64 1, align 4
// CXX_LLVM: %[[B_ADDR:.*]] = alloca { i32, i32 }, i64 1, align 4
// CXX_LLVM: %[[C_ADDR:.*]] = alloca { i32, i32 }, i64 1, align 4
// CXX_LLVM: %[[TMP_A:.*]] = load { i32, i32 }, ptr %[[A_ADDR]], align 4
// CXX_LLVM: %[[TMP_B:.*]] = load { i32, i32 }, ptr %[[B_ADDR]], align 4
// CXX_LLVM: %[[B_REAL:.*]] = extractvalue { i32, i32 } %[[TMP_B]], 0
// CXX_LLVM: %[[B_IMAG:.*]] = extractvalue { i32, i32 } %[[TMP_B]], 1
// CXX_LLVM: %[[A_REAL:.*]] = extractvalue { i32, i32 } %[[TMP_A]], 0
// CXX_LLVM: %[[A_IMAG:.*]] = extractvalue { i32, i32 } %[[TMP_A]], 1
// CXX_LLVM: %[[ADD_REAL:.*]] = add i32 %[[B_REAL]], %[[A_REAL]]
// CXX_LLVM: %[[ADD_IMAG:.*]] = add i32 %[[B_IMAG]], %[[A_IMAG]]
// CXX_LLVM: %[[TMP_RESULT:.*]] = insertvalue { i32, i32 } poison, i32 %[[ADD_REAL]], 0
// CXX_LLVM: %[[RESULT:.*]] = insertvalue { i32, i32 } %[[TMP_RESULT]], i32 %[[ADD_IMAG]], 1
// CXX_LLVM: store { i32, i32 } %[[RESULT]], ptr %[[B_ADDR]], align 4
// CXX_LLVM: %[[TMP_B:.*]] = load { i32, i32 }, ptr %[[B_ADDR]], align 4
// CXX_LLVM: store { i32, i32 } %[[TMP_B]], ptr %[[C_ADDR]], align 4

// CXX_OGCG: %[[A_ADDR:.*]] = alloca { i32, i32 }, align 4
// CXX_OGCG: %[[B_ADDR:.*]] = alloca { i32, i32 }, align 4
// CXX_OGCG: %[[C_ADDR:.*]] = alloca { i32, i32 }, align 4
// CXX_OGCG: %a.realp = getelementptr inbounds nuw { i32, i32 }, ptr %[[A_ADDR]], i32 0, i32 0
// CXX_OGCG: %a.real = load volatile i32, ptr %a.realp, align 4
// CXX_OGCG: %a.imagp = getelementptr inbounds nuw { i32, i32 }, ptr %[[A_ADDR]], i32 0, i32 1
// CXX_OGCG: %a.imag = load volatile i32, ptr %a.imagp, align 4
// CXX_OGCG: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[B_ADDR]], i32 0, i32 0
// CXX_OGCG: %[[B_REAL:.*]] = load volatile i32, ptr %[[B_REAL_PTR]], align 4
// CXX_OGCG: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[B_ADDR]], i32 0, i32 1
// CXX_OGCG: %[[B_IMAG:.*]] = load volatile i32, ptr %[[B_IMAG_PTR]], align 4
// CXX_OGCG: %[[ADD_REAL:.*]] = add i32 %[[B_REAL]], %[[A_REAL]]
// CXX_OGCG: %[[ADD_IMAG:.*]] = add i32 %[[B_IMAG]], %[[A_IMAG]]
// CXX_OGCG: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[B_ADDR]], i32 0, i32 0
// CXX_OGCG: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[B_ADDR]], i32 0, i32 1
// CXX_OGCG: store volatile i32 %[[ADD_REAL]], ptr %[[B_REAL_PTR]], align 4
// CXX_OGCG: store volatile i32 %[[ADD_IMAG]], ptr %[[B_IMAG_PTR]], align 4
// CXX_OGCG: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[B_ADDR]], i32 0, i32 0
// CXX_OGCG: %[[B_REAL:.*]] = load volatile i32, ptr %[[B_REAL_PTR]], align 4
// CXX_OGCG: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[B_ADDR]], i32 0, i32 1
// CXX_OGCG: %[[B_IMAG:.*]] = load volatile i32, ptr %[[B_IMAG_PTR]], align 4
// CXX_OGCG: %[[C_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[C_ADDR]], i32 0, i32 0
// CXX_OGCG: %[[C_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[C_ADDR]], i32 0, i32 1
// CXX_OGCG: store i32 %[[B_REAL]], ptr %[[C_REAL_PTR]], align 4
// CXX_OGCG: store i32 %[[B_IMAG]], ptr %[[C_IMAG_PTR]], align 4
