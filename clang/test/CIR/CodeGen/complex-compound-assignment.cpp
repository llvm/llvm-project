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

void foo5() {
  float _Complex a;
  float b;
  a += b;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["b"]
// CIR: %[[TMP_B:.*]] = cir.load{{.*}} %[[B_ADDR]] : !cir.ptr<!cir.float>, !cir.float
// CIR: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: %[[A_REAL:.*]] = cir.complex.real %[[TMP_A]] : !cir.complex<!cir.float> -> !cir.float
// CIR: %[[A_IMAG:.*]] = cir.complex.imag %[[TMP_A]] : !cir.complex<!cir.float> -> !cir.float
// CIR: %[[RESULT_REAL:.*]] = cir.binop(add, %[[A_REAL]], %[[TMP_B]]) : !cir.float
// CIR: %[[RESULT:.*]] = cir.complex.create %[[RESULT_REAL]], %[[A_IMAG]] : !cir.float -> !cir.complex<!cir.float>
// CIR: cir.store{{.*}} %[[RESULT]], %[[A_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// LLVM: %[[A_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[B_ADDR:.*]] = alloca float, i64 1, align 4
// LLVM: %[[TMP_B:.*]] = load float, ptr %[[B_ADDR]], align 4
// LLVM: %[[TMP_A:.*]] = load { float, float }, ptr %[[A_ADDR]], align 4
// LLVM: %[[A_REAL:.*]] = extractvalue { float, float } %[[TMP_A]], 0
// LLVM: %[[A_IMAG:.*]] = extractvalue { float, float } %[[TMP_A]], 1
// LLVM: %[[RESULT_REAL:.*]] = fadd float %[[A_REAL]], %[[TMP_B]]
// LLVM: %[[TMP_RESULT:.*]] = insertvalue { float, float } {{.*}}, float %[[RESULT_REAL]], 0
// LLVM: %[[RESULT:.*]] = insertvalue { float, float } %[[TMP_RESULT]], float %[[A_IMAG]], 1
// LLVM: store { float, float } %[[RESULT]], ptr %[[A_ADDR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[B_ADDR:.*]] = alloca float, align 4
// OGCG: %[[TMP_B:.*]] = load float, ptr %[[B_ADDR]], align 4
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load float, ptr %[[A_REAL_PTR]], align 4
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load float, ptr %[[A_IMAG_PTR]], align 4
// OGCG: %[[ADD_REAL:.*]] = fadd float %[[A_REAL]], %[[TMP_B]]
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: store float %[[ADD_REAL]], ptr %[[A_REAL_PTR]], align 4
// OGCG: store float %[[A_IMAG]], ptr %[[A_IMAG_PTR]], align 4

void foo6() {
  int _Complex a;
  int _Complex b;
  b *= a;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["b"]
// CIR: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR: %[[TMP_B:.*]] = cir.load{{.*}} %[[B_ADDR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR: %[[B_REAL:.*]] = cir.complex.real %[[TMP_B]] : !cir.complex<!s32i> -> !s32i
// CIR: %[[B_IMAG:.*]] = cir.complex.imag %[[TMP_B]] : !cir.complex<!s32i> -> !s32i
// CIR: %[[A_REAL:.*]] = cir.complex.real %[[TMP_A]] : !cir.complex<!s32i> -> !s32i
// CIR: %[[A_IMAG:.*]] = cir.complex.imag %[[TMP_A]] : !cir.complex<!s32i> -> !s32i
// CIR: %[[MUL_BR_AR:.*]] = cir.binop(mul, %[[B_REAL]], %[[A_REAL]]) : !s32i
// CIR: %[[MUL_BI_AI:.*]] = cir.binop(mul, %[[B_IMAG]], %[[A_IMAG]]) : !s32i
// CIR: %[[MUL_BR_AI:.*]] = cir.binop(mul, %[[B_REAL]], %[[A_IMAG]]) : !s32i
// CIR: %[[MUL_BI_AR:.*]] = cir.binop(mul, %[[B_IMAG]], %[[A_REAL]]) : !s32i
// CIR: %[[RESULT_REAL:.*]] = cir.binop(sub, %[[MUL_BR_AR]], %[[MUL_BI_AI]]) : !s32i
// CIR: %[[RESULT_IMAG:.*]] = cir.binop(add, %[[MUL_BR_AI]], %[[MUL_BI_AR]]) : !s32i
// CIR: %[[RESULT:.*]] = cir.complex.create %[[RESULT_REAL]], %[[RESULT_IMAG]] : !s32i -> !cir.complex<!s32i>
// CIR: cir.store{{.*}} %[[RESULT]], %[[B_ADDR]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>

// LLVM: %[[A_ADDR:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[B_ADDR:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[TMP_A:.*]] = load { i32, i32 }, ptr %[[A_ADDR]], align 4
// LLVM: %[[TMP_B:.*]] = load { i32, i32 }, ptr %[[B_ADDR]], align 4
// LLVM: %[[B_REAL:.*]] = extractvalue { i32, i32 } %[[TMP_B]], 0
// LLVM: %[[B_IMAG:.*]] = extractvalue { i32, i32 } %[[TMP_B]], 1
// LLVM: %[[A_REAL:.*]] = extractvalue { i32, i32 } %[[TMP_A]], 0
// LLVM: %[[A_IMAG:.*]] = extractvalue { i32, i32 } %[[TMP_A]], 1
// LLVM: %[[MUL_BR_AR:.*]] = mul i32 %[[B_REAL]], %[[A_REAL]]
// LLVM: %[[MUL_BI_AI:.*]] = mul i32 %[[B_IMAG]], %[[A_IMAG]]
// LLVM: %[[MUL_BR_AI:.*]] = mul i32 %[[B_REAL]], %[[A_IMAG]]
// LLVM: %[[MUL_BI_AR:.*]] = mul i32 %[[B_IMAG]], %[[A_REAL]]
// LLVM: %[[RESULT_REAL:.*]] = sub i32 %[[MUL_BR_AR]], %[[MUL_BI_AI]]
// LLVM: %[[RESULT_IMAG:.*]] = add i32 %[[MUL_BR_AI]], %[[MUL_BI_AR]]
// LLVM: %[[MUL_A_B:.*]] = insertvalue { i32, i32 } {{.*}}, i32 %[[RESULT_REAL]], 0
// LLVM: %[[RESULT:.*]] = insertvalue { i32, i32 } %[[MUL_A_B]], i32 %[[RESULT_IMAG]], 1
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
// OGCG: %[[MUL_BR_AR:.*]] = mul i32 %[[B_REAL]], %[[A_REAL]]
// OGCG: %[[MUL_BI_AI:.*]] = mul i32 %[[B_IMAG]], %[[A_IMAG]]
// OGCG: %[[RESULT_REAL:.*]] = sub i32 %[[MUL_BR_AR]], %[[MUL_BI_AI]]
// OGCG: %[[MUL_BI_AR:.*]] = mul i32 %[[B_IMAG]], %[[A_REAL]]
// OGCG: %[[MUL_BR_AI:.*]] = mul i32 %[[B_REAL]], %[[A_IMAG]]
// OGCG: %[[RESULT_IMAG:.*]] = add i32 %[[MUL_BI_AR]], %[[MUL_BR_AI]]
// OGCG: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG: store i32 %[[RESULT_REAL]], ptr %[[B_REAL_PTR]], align 4
// OGCG: store i32 %[[RESULT_IMAG]], ptr %[[B_IMAG_PTR]], align 4

void foo7() {
  float _Complex a;
  float _Complex b;
  b *= a;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["b"]
// CIR: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: %[[TMP_B:.*]] = cir.load{{.*}} %[[B_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: %[[B_REAL:.*]] = cir.complex.real %[[TMP_B]] : !cir.complex<!cir.float> -> !cir.float
// CIR: %[[B_IMAG:.*]] = cir.complex.imag %[[TMP_B]] : !cir.complex<!cir.float> -> !cir.float
// CIR: %[[A_REAL:.*]] = cir.complex.real %[[TMP_A]] : !cir.complex<!cir.float> -> !cir.float
// CIR: %[[A_IMAG:.*]] = cir.complex.imag %[[TMP_A]] : !cir.complex<!cir.float> -> !cir.float
// CIR: %[[MUL_BR_AR:.*]] = cir.binop(mul, %[[B_REAL]], %[[A_REAL]]) : !cir.float
// CIR: %[[MUL_BI_AI:.*]] = cir.binop(mul, %[[B_IMAG]], %[[A_IMAG]]) : !cir.float
// CIR: %[[MUL_BR_AI:.*]] = cir.binop(mul, %[[B_REAL]], %[[A_IMAG]]) : !cir.float
// CIR: %[[MUL_BI_AR:.*]] = cir.binop(mul, %[[B_IMAG]], %[[A_REAL]]) : !cir.float
// CIR: %[[C_REAL:.*]] = cir.binop(sub, %[[MUL_BR_AR]], %[[MUL_BI_AI]]) : !cir.float
// CIR: %[[C_IMAG:.*]] = cir.binop(add, %[[MUL_BR_AI]], %[[MUL_BI_AR]]) : !cir.float
// CIR: %[[COMPLEX:.*]] = cir.complex.create %[[C_REAL]], %[[C_IMAG]] : !cir.float -> !cir.complex<!cir.float>
// CIR: %[[IS_C_REAL_NAN:.*]] = cir.cmp(ne, %[[C_REAL]], %[[C_REAL]]) : !cir.float, !cir.bool
// CIR: %[[IS_C_IMAG_NAN:.*]] = cir.cmp(ne, %[[C_IMAG]], %[[C_IMAG]]) : !cir.float, !cir.bool
// CIR: %[[CONST_FALSE:.*]] = cir.const #false
// CIR: %[[SELECT_CONDITION:.*]] = cir.select if %[[IS_C_REAL_NAN]] then %[[IS_C_IMAG_NAN]] else %[[CONST_FALSE]] : (!cir.bool, !cir.bool, !cir.bool) -> !cir.bool
// CIR: %[[RESULT:.*]] = cir.ternary(%[[SELECT_CONDITION]], true {
// CIR:   %[[LIBC_COMPLEX:.*]] = cir.call @__mulsc3(%[[B_REAL]], %[[B_IMAG]], %[[A_REAL]], %[[A_IMAG]]) : (!cir.float, !cir.float, !cir.float, !cir.float) -> !cir.complex<!cir.float>
// CIR:   cir.yield %[[LIBC_COMPLEX]] : !cir.complex<!cir.float>
// CIR: }, false {
// CIR:   cir.yield %[[COMPLEX]] : !cir.complex<!cir.float>
// CIR: }) : (!cir.bool) -> !cir.complex<!cir.float>
// CIR: cir.store{{.*}} %[[RESULT]], %[[B_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// LLVM: %[[A_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[B_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[TMP_A:.*]] = load { float, float }, ptr %[[A_ADDR]], align 4
// LLVM: %[[TMP_B:.*]] = load { float, float }, ptr %[[B_ADDR]], align 4
// LLVM: %[[B_REAL:.*]] = extractvalue { float, float } %[[TMP_B]], 0
// LLVM: %[[B_IMAG:.*]] = extractvalue { float, float } %[[TMP_B]], 1
// LLVM: %[[A_REAL:.*]] = extractvalue { float, float } %[[TMP_A]], 0
// LLVM: %[[A_IMAG:.*]] = extractvalue { float, float } %[[TMP_A]], 1
// LLVM: %[[MUL_BR_AR:.*]] = fmul float %[[B_REAL]], %[[A_REAL]]
// LLVM: %[[MUL_BI_AI:.*]] = fmul float %[[B_IMAG]], %[[A_IMAG]]
// LLVM: %[[MUL_BR_AI:.*]] = fmul float %[[B_REAL]], %[[A_IMAG]]
// LLVM: %[[MUL_BI_AR:.*]] = fmul float %[[B_IMAG]], %[[A_REAL]]
// LLVM: %[[C_REAL:.*]] = fsub float %[[MUL_BR_AR]], %[[MUL_BI_AI]]
// LLVM: %[[C_IMAG:.*]] = fadd float %[[MUL_BR_AI]], %[[MUL_BI_AR]]
// LLVM: %[[MUL_A_B:.*]] = insertvalue { float, float } {{.*}}, float %[[C_REAL]], 0
// LLVM: %[[COMPLEX:.*]] = insertvalue { float, float } %[[MUL_A_B]], float %[[C_IMAG]], 1
// LLVM: %[[IS_C_REAL_NAN:.*]] = fcmp une float %[[C_REAL]], %[[C_REAL]]
// LLVM: %[[IS_C_IMAG_NAN:.*]] = fcmp une float %[[C_IMAG]], %[[C_IMAG]]
// LLVM: %[[SELECT_CONDITION:.*]] = and i1 %[[IS_C_REAL_NAN]], %[[IS_C_IMAG_NAN]]
// LLVM: br i1 %[[SELECT_CONDITION]], label %[[THEN_LABEL:.*]], label %[[ELSE_LABEL:.*]]
// LLVM: [[THEN_LABEL]]:
// LLVM:  %[[LIBC_COMPLEX:.*]] = call { float, float } @__mulsc3(float %[[B_REAL]], float %[[B_IMAG]], float %[[A_REAL]], float %[[A_IMAG]])
// LLVM:  br label %[[PHI_BRANCH:.*]]
// LLVM: [[ELSE_LABEL]]:
// LLVM:  br label %[[PHI_BRANCH:]]
// LLVM: [[PHI_BRANCH:]]:
// LLVM:  %[[RESULT:.*]] = phi { float, float } [ %[[COMPLEX]], %[[ELSE_LABEL]] ], [ %[[LIBC_COMPLEX]], %[[THEN_LABEL]] ]
// LLVM:  br label %[[END_LABEL:.*]]
// LLVM: [[END_LABEL]]:
// LLVM:  store { float, float } %[[RESULT]], ptr %[[B_ADDR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[B_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[COMPLEX_CALL_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load float, ptr %[[A_REAL_PTR]], align 4
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load float, ptr %[[A_IMAG_PTR]], align 4
// OGCG: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG: %[[B_REAL:.*]] = load float, ptr %[[B_REAL_PTR]], align 4
// OGCG: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG: %[[B_IMAG:.*]] = load float, ptr %[[B_IMAG_PTR]], align 4
// OGCG: %[[MUL_BR_AR:.*]] = fmul float %[[B_REAL]], %[[A_REAL]]
// OGCG: %[[MUL_BI_AI:.*]] = fmul float %[[B_IMAG]], %[[A_IMAG]]
// OGCG: %[[MUL_BR_AI:.*]] = fmul float %[[B_REAL]], %[[A_IMAG]]
// OGCG: %[[MUL_BI_AR:.*]] = fmul float %[[B_IMAG]], %[[A_REAL]]
// OGCG: %[[C_REAL:.*]] = fsub float %[[MUL_BR_AR]], %[[MUL_BI_AI]]
// OGCG: %[[C_IMAG:.*]] = fadd float %[[MUL_BR_AI]], %[[MUL_BI_AR]]
// OGCG: %[[IS_C_REAL_NAN:.*]] = fcmp uno float %[[C_REAL]], %[[C_REAL]]
// OGCG: br i1 %[[IS_C_REAL_NAN]], label %[[COMPLEX_IS_IMAG_NAN:.*]], label %[[END_LABEL:.*]], !prof !2
// OGCG: [[COMPLEX_IS_IMAG_NAN]]:
// OGCG:  %[[IS_C_IMAG_NAN:.*]] = fcmp uno float %[[C_IMAG]], %[[C_IMAG]]
// OGCG:  br i1 %[[IS_C_IMAG_NAN]], label %[[COMPLEX_LIB_CALL:.*]], label %[[END_LABEL]], !prof !2
// OGCG: [[COMPLEX_LIB_CALL]]:
// OGCG:  %[[CALL_RESULT:.*]] = call{{.*}} <2 x float> @__mulsc3(float noundef %[[B_REAL]], float noundef %[[B_IMAG]], float noundef %[[A_REAL]], float noundef %[[A_IMAG]])
// OGCG:  store <2 x float> %[[CALL_RESULT]], ptr %[[COMPLEX_CALL_ADDR]], align 4
// OGCG:  %[[COMPLEX_CALL_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[COMPLEX_CALL_ADDR]], i32 0, i32 0
// OGCG:  %[[COMPLEX_CALL_REAL:.*]] = load float, ptr %[[COMPLEX_CALL_REAL_PTR]], align 4
// OGCG:  %[[COMPLEX_CALL_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[COMPLEX_CALL_ADDR]], i32 0, i32 1
// OGCG:  %[[COMPLEX_CALL_IMAG:.*]] = load float, ptr %[[COMPLEX_CALL_IMAG_PTR]], align 4
// OGCG:  br label %[[END_LABEL]]
// OGCG: [[END_LABEL]]:
// OGCG:  %[[FINAL_REAL:.*]] = phi float [ %[[C_REAL]], %[[ENTRY:.*]] ], [ %[[C_REAL]], %[[COMPLEX_IS_IMAG_NAN]] ], [ %[[COMPLEX_CALL_REAL]], %[[COMPLEX_LIB_CALL]] ]
// OGCG:  %[[FINAL_IMAG:.*]] = phi float [ %[[C_IMAG]], %[[ENTRY]] ], [ %[[C_IMAG]], %[[COMPLEX_IS_IMAG_NAN]] ], [ %[[COMPLEX_CALL_IMAG]], %[[COMPLEX_LIB_CALL]] ]
// OGCG:  %[[C_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG:  %[[C_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG:  store float %[[FINAL_REAL]], ptr %[[C_REAL_PTR]], align 4
// OGCG:  store float %[[FINAL_IMAG]], ptr %[[C_IMAG_PTR]], align 4

void foo8() {
  float _Complex a;
  float b;
  a *= b;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["b"]
// CIR: %[[TMP_B:.*]] = cir.load{{.*}} %[[B_ADDR]] : !cir.ptr<!cir.float>, !cir.float
// CIR: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: %[[A_REAL:.*]] = cir.complex.real %[[TMP_A]] : !cir.complex<!cir.float> -> !cir.float
// CIR: %[[A_IMAG:.*]] = cir.complex.imag %[[TMP_A]] : !cir.complex<!cir.float> -> !cir.float
// CIR: %[[RESULT_REAL:.*]] = cir.binop(mul, %[[A_REAL]], %[[TMP_B]]) : !cir.float
// CIR: %[[RESULT_IMAG:.*]] = cir.binop(mul, %[[A_IMAG]], %[[TMP_B]]) : !cir.float
// CIR: %[[RESULT:.*]] = cir.complex.create %[[RESULT_REAL]], %[[RESULT_IMAG]] : !cir.float -> !cir.complex<!cir.float>
// CIR: cir.store{{.*}} %[[RESULT]], %[[A_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// LLVM: %[[A_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[B_ADDR:.*]] = alloca float, i64 1, align 4
// LLVM: %[[TMP_B:.*]] = load float, ptr %[[B_ADDR]], align 4
// LLVM: %[[TMP_A:.*]] = load { float, float }, ptr %[[A_ADDR]], align 4
// LLVM: %[[A_REAL:.*]] = extractvalue { float, float } %[[TMP_A]], 0
// LLVM: %[[A_IMAG:.*]] = extractvalue { float, float } %[[TMP_A]], 1
// LLVM: %[[RESULT_REAL:.*]] = fmul float %[[A_REAL]], %[[TMP_B]]
// LLVM: %[[RESULT_IMAG:.*]] = fmul float %[[A_IMAG]], %[[TMP_B]]
// LLVM: %[[TMP_RESULT:.*]] = insertvalue { float, float } {{.*}}, float %[[RESULT_REAL]], 0
// LLVM: %[[RESULT:.*]] = insertvalue { float, float } %[[TMP_RESULT]], float %[[RESULT_IMAG]], 1
// LLVM: store { float, float } %[[RESULT]], ptr %[[A_ADDR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[B_ADDR:.*]] = alloca float, align 4
// OGCG: %[[TMP_B:.*]] = load float, ptr %[[B_ADDR]], align 4
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load float, ptr %[[A_REAL_PTR]], align 4
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load float, ptr %[[A_IMAG_PTR]], align 4
// OGCG: %[[RESULT_REAL:.*]] = fmul float %[[A_REAL]], %[[TMP_B]]
// OGCG: %[[RESULT_IMAG:.*]] = fmul float %[[A_IMAG]], %[[TMP_B]]
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: store float %[[RESULT_REAL]], ptr %[[A_REAL_PTR]], align 4
// OGCG: store float %[[RESULT_IMAG]], ptr %[[A_IMAG_PTR]], align 4

void foo10() {
  float _Complex a;
  float _Complex b;
  a /= b;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["b"]
// CIR: %[[TMP_B:.*]] = cir.load{{.*}} %[[B_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: %[[A_REAL:.*]] = cir.complex.real %[[TMP_A]] : !cir.complex<!cir.float> -> !cir.float
// CIR: %[[A_IMAG:.*]] = cir.complex.imag %[[TMP_A]] : !cir.complex<!cir.float> -> !cir.float
// CIR: %[[B_REAL:.*]] = cir.complex.real %[[TMP_B]] : !cir.complex<!cir.float> -> !cir.float
// CIR: %[[B_IMAG:.*]] = cir.complex.imag %[[TMP_B]] : !cir.complex<!cir.float> -> !cir.float
// CIR: %[[RESULT:.*]] = cir.call @__divsc3(%[[A_REAL]], %[[A_IMAG]], %[[B_REAL]], %[[B_IMAG]]) : (!cir.float, !cir.float, !cir.float, !cir.float) -> !cir.complex<!cir.float>
// CIR: cir.store{{.*}} %[[RESULT]], %[[A_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// LLVM: %[[A_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[B_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[TMP_B:.*]] = load { float, float }, ptr %[[B_ADDR]], align 4
// LLVM: %[[TMP_A:.*]] = load { float, float }, ptr %[[A_ADDR]], align 4
// LLVM: %[[A_REAL:.*]] = extractvalue { float, float } %[[TMP_A]], 0
// LLVM: %[[A_IMAG:.*]] = extractvalue { float, float } %[[TMP_A]], 1
// LLVM: %[[B_REAL:.*]] = extractvalue { float, float } %[[TMP_B]], 0
// LLVM: %[[B_IMAG:.*]] = extractvalue { float, float } %[[TMP_B]], 1
// LLVM: %[[RESULT:.*]] = call { float, float } @__divsc3(float %[[A_REAL]], float %[[A_IMAG]], float %[[B_REAL]], float %[[B_IMAG]])
// LLVM: store { float, float } %[[RESULT]], ptr %[[A_ADDR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[B_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[RESULT_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG: %[[B_REAL:.*]] = load float, ptr %[[B_REAL_PTR]], align 4
// OGCG: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG: %[[B_IMAG:.*]] = load float, ptr %[[B_IMAG_PTR]], align 4
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load float, ptr %[[A_REAL_PTR]], align 4
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load float, ptr %[[A_IMAG_PTR]], align 4
// OGCG: %[[RESULT:.*]] = call{{.*}} <2 x float> @__divsc3(float noundef %[[A_REAL]], float noundef %[[A_IMAG]], float noundef %[[B_REAL]], float noundef %[[B_IMAG]])
// OGCG: store <2 x float> %[[RESULT]], ptr %[[RESULT_ADDR]], align 4
// OGCG: %[[RESULT_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[RESULT_ADDR]], i32 0, i32 0
// OGCG: %[[RESULT_REAL:.*]] = load float, ptr %[[RESULT_REAL_PTR]], align 4
// OGCG: %[[RESULT_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[RESULT_ADDR]], i32 0, i32 1
// OGCG: %[[RESULT_IMAG:.*]] = load float, ptr %[[RESULT_IMAG_PTR]], align 4
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: store float %[[RESULT_REAL]], ptr %[[A_REAL_PTR]], align 4
// OGCG: store float %[[RESULT_IMAG]], ptr %[[A_IMAG_PTR]], align 4

void foo11() {
  float _Complex a;
  float b;
  a /= b;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["b"]
// CIR: %[[TMP_B:.*]] = cir.load{{.*}} %[[B_ADDR]] : !cir.ptr<!cir.float>, !cir.float
// CIR: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: %[[A_REAL:.*]] = cir.complex.real %[[TMP_A]] : !cir.complex<!cir.float> -> !cir.float
// CIR: %[[A_IMAG:.*]] = cir.complex.imag %[[TMP_A]] : !cir.complex<!cir.float> -> !cir.float
// CIR: %[[RESULT_REAL:.*]] = cir.binop(div, %[[A_REAL]], %[[TMP_B]]) : !cir.float
// CIR: %[[RESULT_IMAG:.*]] = cir.binop(div, %[[A_IMAG]], %[[TMP_B]]) : !cir.float
// CIR: %[[RESULT:.*]] = cir.complex.create %[[RESULT_REAL]], %[[RESULT_IMAG]] : !cir.float -> !cir.complex<!cir.float>
// CIR: cir.store{{.*}} %[[RESULT]], %[[A_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// LLVM: %[[A_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[B_ADDR:.*]] = alloca float, i64 1, align 4
// LLVM: %[[TMP_B:.*]] = load float, ptr %[[B_ADDR]], align 4
// LLVM: %[[TMP_A:.*]] = load { float, float }, ptr %[[A_ADDR]], align 4
// LLVM: %[[A_REAL:.*]] = extractvalue { float, float } %[[TMP_A]], 0
// LLVM: %[[A_IMAG:.*]] = extractvalue { float, float } %[[TMP_A]], 1
// LLVM: %[[RESULT_REAL:.*]] = fdiv float %[[A_REAL]], %[[TMP_B]]
// LLVM: %[[RESULT_IMAG:.*]] = fdiv float %[[A_IMAG]], %[[TMP_B]]
// LLVM: %[[TMP_RESULT:.*]] = insertvalue { float, float } {{.*}}, float %[[RESULT_REAL]], 0
// LLVM: %[[RESULT:.*]] = insertvalue { float, float } %[[TMP_RESULT]], float %[[RESULT_IMAG]], 1
// LLVM: store { float, float } %[[RESULT]], ptr %[[A_ADDR]], align 4

// OGCG:  %[[A_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[B_ADDR:.*]] = alloca float, align 4
// OGCG: %[[TMP_B:.*]] = load float, ptr %[[B_ADDR]], align 4
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load float, ptr %[[A_REAL_PTR]], align 4
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load float, ptr %[[A_IMAG_PTR]], align 4
// OGCG: %[[RESULT_REAL:.*]] = fdiv float %[[A_REAL]], %[[TMP_B]]
// OGCG: %[[RESULT_IMAG:.*]] = fdiv float %[[A_IMAG]], %[[TMP_B]]
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: store float %[[RESULT_REAL]], ptr %[[A_REAL_PTR]], align 4
// OGCG: store float %[[RESULT_IMAG]], ptr %[[A_IMAG_PTR]], align 4

void foo12() {
  int _Complex  a;
  int b;
  a /= b;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["b"]
// CIR: %[[TMP_B:.*]] = cir.load{{.*}} %[[B_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR: %[[CONST_0:.*]] = cir.const #cir.int<0> : !s32i
// CIR: %[[B_COMPLEX:.*]] = cir.complex.create %[[TMP_B]], %[[CONST_0]] : !s32i -> !cir.complex<!s32i>
// CIR: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR: %[[A_REAL:.*]] = cir.complex.real %[[TMP_A]] : !cir.complex<!s32i> -> !s32i
// CIR: %[[A_IMAG:.*]] = cir.complex.imag %[[TMP_A]] : !cir.complex<!s32i> -> !s32i
// CIR: %[[B_REAL:.*]] = cir.complex.real %[[B_COMPLEX]] : !cir.complex<!s32i> -> !s32i
// CIR: %[[B_IMAG:.*]] = cir.complex.imag %[[B_COMPLEX]] : !cir.complex<!s32i> -> !s32i
// CIR: %[[MUL_AR_BR:.*]] = cir.binop(mul, %[[A_REAL]], %[[B_REAL]]) : !s32i
// CIR: %[[MUL_AI_BI:.*]] = cir.binop(mul, %[[A_IMAG]], %[[B_IMAG]]) : !s32i
// CIR: %[[MUL_BR_BR:.*]] = cir.binop(mul, %[[B_REAL]], %[[B_REAL]]) : !s32i
// CIR: %[[MUL_BI_BI:.*]] = cir.binop(mul, %[[B_IMAG]], %[[B_IMAG]]) : !s32i
// CIR: %[[ADD_ARBR_AIBI:.*]] = cir.binop(add, %[[MUL_AR_BR]], %[[MUL_AI_BI]]) : !s32i
// CIR: %[[ADD_BRBR_BIBI:.*]] = cir.binop(add, %[[MUL_BR_BR]], %[[MUL_BI_BI]]) : !s32i
// CIR: %[[RESULT_REAL:.*]] = cir.binop(div, %[[ADD_ARBR_AIBI]], %[[ADD_BRBR_BIBI]]) : !s32i
// CIR: %[[MUL_AI_BR:.*]] = cir.binop(mul, %[[A_IMAG]], %[[B_REAL]]) : !s32i
// CIR: %[[MUL_AR_BI:.*]] = cir.binop(mul, %[[A_REAL]], %[[B_IMAG]]) : !s32i
// CIR: %[[SUB_AIBR_ARBI:.*]] = cir.binop(sub, %[[MUL_AI_BR]], %[[MUL_AR_BI]]) : !s32i
// CIR: %[[RESULT_IMAG:.*]] = cir.binop(div, %[[SUB_AIBR_ARBI]], %[[ADD_BRBR_BIBI]]) : !s32i
// CIR: %[[RESULT:.*]] = cir.complex.create %[[RESULT_REAL]], %[[RESULT_IMAG]] : !s32i -> !cir.complex<!s32i>
// CIR: cir.store{{.*}} %[[RESULT]], %[[A_ADDR]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>

// LLVM: %[[A_ADDR:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[B_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM: %[[TMP_B:.*]] = load i32, ptr %[[B_ADDR]], align 4
// LLVM: %[[TMP_B_COMPLEX:.*]] = insertvalue { i32, i32 } {{.*}}, i32 %[[TMP_B]], 0
// LLVM: %[[B_COMPLEX:.*]] = insertvalue { i32, i32 } %[[TMP_B_COMPLEX]], i32 0, 1
// LLVM: %[[TMP_A:.*]] = load { i32, i32 }, ptr %[[A_ADDR]], align 4
// LLVM: %[[A_REAL:.*]] = extractvalue { i32, i32 } %[[TMP_A]], 0
// LLVM: %[[A_IMAG:.*]] = extractvalue { i32, i32 } %[[TMP_A]], 1
// LLVM: %[[MUL_AR_BR:.*]] = mul i32 %[[A_REAL]], %[[TMP_B]]
// LLVM: %[[MUL_AI_BI:.*]] = mul i32 %[[A_IMAG]], 0
// LLVM: %[[MUL_BR_BR:.*]] = mul i32 %[[TMP_B]], %[[TMP_B]]
// LLVM: %[[ADD_ARBR_AIBI:.*]] = add i32 %[[MUL_AR_BR]], %[[MUL_AI_BI]]
// LLVM: %[[ADD_BRBR_BIBI:.*]] = add i32 %[[MUL_BR_BR]], 0
// LLVM: %[[RESULT_REAL:.*]] = sdiv i32 %[[ADD_ARBR_AIBI]], %[[ADD_BRBR_BIBI]]
// LLVM: %[[MUL_AI_BR:.*]] = mul i32 %[[A_IMAG]], %[[TMP_B]]
// LLVM: %[[MUL_AR_BI:.*]] = mul i32 %[[A_REAL]], 0
// LLVM: %[[SUB_AIBR_ARBI:.*]] = sub i32 %[[MUL_AI_BR]], %[[MUL_AR_BI]]
// LLVM: %[[RESULT_IMAG:.*]] = sdiv i32 %[[SUB_AIBR_ARBI]], %[[ADD_BRBR_BIBI]]
// LLVM: %[[TMP_RESULT:.*]] = insertvalue { i32, i32 } {{.*}}, i32 %[[RESULT_REAL]], 0
// LLVM: %[[RESULT:.*]] = insertvalue { i32, i32 } %[[TMP_RESULT]], i32 %[[RESULT_IMAG]], 1
// LLVM: store { i32, i32 } %[[RESULT]], ptr %[[A_ADDR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[B_ADDR:.*]] = alloca i32, align 4
// OGCG: %[[TMP_B:.*]] = load i32, ptr %[[B_ADDR]], align 4
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load i32, ptr %[[A_REAL_PTR]], align 4
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load i32, ptr %[[A_IMAG_PTR]], align 4
// OGCG: %[[MUL_AR_BR:.*]] = mul i32 %[[A_REAL]], %[[TMP_B]]
// OGCG: %[[MUL_AI_BI:.*]] = mul i32 %[[A_IMAG]], 0
// OGCG: %[[ADD_ARBR_AIBI:.*]] = add i32 %[[MUL_AR_BR]], %[[MUL_AI_BI]]
// OGCG: %[[MUL_BR_BR:.*]] = mul i32 %[[TMP_B]], %[[TMP_B]]
// OGCG: %[[ADD_BRBR_BIBI:.*]] = add i32 %[[MUL_BR_BR]], 0
// OGCG: %[[MUL_AI_BR:.*]] = mul i32 %[[A_IMAG]], %[[TMP_B]]
// OGCG: %[[MUL_AR_BI:.*]] = mul i32 %[[A_REAL]], 0
// OGCG: %[[SUB_AIBR_ARBI:.*]] = sub i32 %[[MUL_AI_BR]], %[[MUL_AR_BI]]
// OGCG: %[[RESULT_REAL:.*]] = sdiv i32 %[[ADD_ARBR_AIBI]], %[[ADD_BRBR_BIBI]]
// OGCG: %[[RESULT_IMAG:.*]] = sdiv i32 %[[SUB_AIBR_ARBI]], %[[ADD_BRBR_BIBI]]
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: store i32 %[[RESULT_REAL]], ptr %[[A_REAL_PTR]], align 4
// OGCG: store i32 %[[RESULT_IMAG]], ptr %[[A_IMAG_PTR]], align 4

#ifndef __cplusplus
void foo9() {
  float _Complex a;
  float b;
  b += a;
}
#endif

// C_CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// C_CIR: %[[B_ADDR:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["b"]
// C_CIR: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// C_CIR: %[[TMP_B:.*]] = cir.load{{.*}} %[[B_ADDR]] : !cir.ptr<!cir.float>, !cir.float
// C_CIR: %[[A_REAL:.*]] = cir.complex.real %[[A_ADDR]] : !cir.complex<!cir.float> -> !cir.float
// C_CIR: %[[A_IMAG:.*]] = cir.complex.imag %[[A_ADDR]] : !cir.complex<!cir.float> -> !cir.float
// C_CIR: %[[NEW_REAL:.*]] = cir.binop(add, %[[TMP_B]], %[[A_REAL]]) : !cir.float
// C_CIR: %[[RESULT:.*]] = cir.complex.create %[[NEW_REAL]], %[[A_IMAG]] : !cir.float -> !cir.complex<!cir.float>
// C_CIR: %[[RESULT_REAL:.*]] = cir.complex.real %[[RESULT]] : !cir.complex<!cir.float> -> !cir.float
// C_CIR: cir.store{{.*}} %[[RESULT_REAL]], %[[B_ADDR]] : !cir.float, !cir.ptr<!cir.float>

// C_LLVM: %[[A_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// C_LLVM: %[[B_ADDR:.*]] = alloca float, i64 1, align 4
// C_LLVM: %[[TMP_A:.*]] = load { float, float }, ptr %[[A_ADDR]], align 4
// C_LLVM: %[[TMP_B:.*]] = load float, ptr %[[B_ADDR]], align 4
// C_LLVM: %[[A_REAL:.*]] = extractvalue { float, float } %[[TMP_A]], 0
// C_LLVM: %[[A_IMAG:.*]] = extractvalue { float, float } %[[TMP_A]], 1
// C_LLVM: %[[NEW_REAL:.*]] = fadd float %[[TMP_B]], %[[A_REAL]]
// C_LLVM: %[[TMP_RESULT:.*]] = insertvalue { float, float } {{.*}}, float %[[NEW_REAL]], 0
// C_LLVM: store float %[[NEW_REAL]], ptr %[[B_ADDR]], align 4

// C_OGCG: %[[A_ADDR:.*]] = alloca { float, float }, align 4
// C_OGCG: %[[B_ADDR:.*]] = alloca float, align 4
// C_OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// C_OGCG: %[[A_REAL:.*]] = load float, ptr %[[A_REAL_PTR]], align 4
// C_OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// C_OGCG: %[[A_IMAG:.*]] = load float, ptr %[[A_IMAG_PTR]], align 4
// C_OGCG: %[[TMP_B:.*]] = load float, ptr %[[B_ADDR]], align 4
// C_OGCG: %[[ADD_REAL:.*]] = fadd float %[[TMP_B]], %[[A_REAL]]
// C_OGCG: store float %[[ADD_REAL]], ptr %[[B_ADDR]], align 4