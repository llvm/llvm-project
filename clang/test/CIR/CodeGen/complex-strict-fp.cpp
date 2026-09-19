// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -ffp-contract=on -fexperimental-strict-floating-point -ffp-exception-behavior=strict -emit-cir %s -o %t-strict.cir
// RUN: FileCheck --input-file=%t-strict.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -ffp-contract=on -fexperimental-strict-floating-point -ffp-exception-behavior=strict -emit-llvm %s -o %t-strict.ll
// RUN: FileCheck --input-file=%t-strict.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -ffp-contract=on -fexperimental-strict-floating-point -ffp-exception-behavior=strict -emit-llvm %s -o %t-strict-ogcg.ll
// RUN: FileCheck --input-file=%t-strict-ogcg.ll %s -check-prefix=OGCG

void complex_add() {
  _Complex float a;
  _Complex float b;
  _Complex float c = a + b;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!cir.complex<!cir.float>>
// CIR: %[[B_ADDR:.*]] = cir.alloca "b" {{.*}} : !cir.ptr<!cir.complex<!cir.float>>
// CIR: %[[C_ADDR:.*]] = cir.alloca "c" {{.*}} init : !cir.ptr<!cir.complex<!cir.float>>
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: %[[TMP_B:.*]] = cir.load {{.*}} %[[B_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: %[[RESULT:.*]] = cir.complex.add %[[TMP_A]], %[[TMP_B]] : !cir.complex<!cir.float> {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR: cir.store {{.*}} %[[RESULT]], %[[C_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// LLVM: %[[A_ADDR:.*]] = alloca { float, float }, align 4
// LLVM: %[[B_ADDR:.*]] = alloca { float, float }, align 4
// LLVM: %[[C_ADDR:.*]] = alloca { float, float }, align 4
// LLVM: %[[TMP_A:.*]] = load { float, float }, ptr %[[A_ADDR]], align 4
// LLVM: %[[TMP_B:.*]] = load { float, float }, ptr %[[B_ADDR]], align 4
// LLVM: %[[A_REAL:.*]] = extractvalue { float, float } %[[TMP_A]], 0
// LLVM: %[[A_IMAG:.*]] = extractvalue { float, float } %[[TMP_A]], 1
// LLVM: %[[B_REAL:.*]] = extractvalue { float, float } %[[TMP_B]], 0
// LLVM: %[[B_IMAG:.*]] = extractvalue { float, float } %[[TMP_B]], 1
// LLVM: %[[RESULT_REAL:.*]] = call float @llvm.experimental.constrained.fadd.f32(float %[[A_REAL]], float %[[B_REAL]], metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM: %[[RESULT_IMAG:.*]] = call float @llvm.experimental.constrained.fadd.f32(float %[[A_IMAG]], float %[[B_IMAG]], metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM: %[[TMP_RESULT:.*]] = insertvalue { float, float } poison, float %[[RESULT_REAL]], 0
// LLVM: %[[RESULT:.*]] = insertvalue { float, float } %[[TMP_RESULT]], float %[[RESULT_IMAG]], 1
// LLVM: store { float, float } %[[RESULT]], ptr %[[C_ADDR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[B_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[C_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load float, ptr %[[A_REAL_PTR]], align 4
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load float, ptr %[[A_IMAG_PTR]], align 4
// OGCG: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG: %[[B_REAL:.*]] = load float, ptr %[[B_REAL_PTR]], align 4
// OGCG: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG: %[[B_IMAG:.*]] = load float, ptr %[[B_IMAG_PTR]], align 4
// OGCG: %[[RESULT_REAL:.*]] = call float @llvm.experimental.constrained.fadd.f32(float %[[A_REAL]], float %[[B_REAL]], metadata !"round.tonearest", metadata !"fpexcept.strict") #2
// OGCG: %[[RESULT_IMAG:.*]] = call float @llvm.experimental.constrained.fadd.f32(float %[[A_IMAG]], float %[[B_IMAG]], metadata !"round.tonearest", metadata !"fpexcept.strict") #2
// OGCG: %[[C_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[C_ADDR]], i32 0, i32 0
// OGCG: %[[C_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[C_ADDR]], i32 0, i32 1
// OGCG: store float %[[RESULT_REAL]], ptr %[[C_REAL_PTR]], align 4
// OGCG: store float %[[RESULT_IMAG]], ptr %[[C_IMAG_PTR]], align 4

void complex_sub() {
  _Complex float a;
  _Complex float b;
  _Complex float c = a - b;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!cir.complex<!cir.float>>
// CIR: %[[B_ADDR:.*]] = cir.alloca "b" {{.*}} : !cir.ptr<!cir.complex<!cir.float>>
// CIR: %[[C_ADDR:.*]] = cir.alloca "c" {{.*}} init : !cir.ptr<!cir.complex<!cir.float>>
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: %[[TMP_B:.*]] = cir.load {{.*}} %[[B_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: %[[RESULT:.*]] = cir.complex.sub %[[TMP_A]], %[[TMP_B]] : !cir.complex<!cir.float> {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
// CIR: cir.store {{.*}} %[[RESULT]], %[[C_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// LLVM: %[[A_ADDR:.*]] = alloca { float, float }, align 4
// LLVM: %[[B_ADDR:.*]] = alloca { float, float }, align 4
// LLVM: %[[C_ADDR:.*]] = alloca { float, float }, align 4
// LLVM: %[[TMP_A:.*]] = load { float, float }, ptr %[[A_ADDR]], align 4
// LLVM: %[[TMP_B:.*]] = load { float, float }, ptr %[[B_ADDR]], align 4
// LLVM: %[[A_REAL:.*]] = extractvalue { float, float } %[[TMP_A]], 0
// LLVM: %[[A_IMAG:.*]] = extractvalue { float, float } %[[TMP_A]], 1
// LLVM: %[[B_REAL:.*]] = extractvalue { float, float } %[[TMP_B]], 0
// LLVM: %[[B_IMAG:.*]] = extractvalue { float, float } %[[TMP_B]], 1
// LLVM: %[[RESULT_REAL:.*]] = call float @llvm.experimental.constrained.fsub.f32(float %[[A_REAL]], float %[[B_REAL]], metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM: %[[RESULT_IMAG:.*]] = call float @llvm.experimental.constrained.fsub.f32(float %[[A_IMAG]], float %[[B_IMAG]], metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM: %[[TMP_RESULT:.*]] = insertvalue { float, float } poison, float %[[RESULT_REAL]], 0
// LLVM: %[[RESULT:.*]] = insertvalue { float, float } %[[TMP_RESULT]], float %[[RESULT_IMAG]], 1
// LLVM: store { float, float } %[[RESULT]], ptr %[[C_ADDR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[B_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[C_ADDR:.*]] = alloca { float, float }, align 4
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load float, ptr %[[A_REAL_PTR]], align 4
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load float, ptr %[[A_IMAG_PTR]], align 4
// OGCG: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG: %[[B_REAL:.*]] = load float, ptr %[[B_REAL_PTR]], align 4
// OGCG: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG: %[[B_IMAG:.*]] = load float, ptr %[[B_IMAG_PTR]], align 4
// OGCG: %[[RESULT_REAL:.*]] = call float @llvm.experimental.constrained.fsub.f32(float %[[A_REAL]], float %[[B_REAL]], metadata !"round.tonearest", metadata !"fpexcept.strict") #2
// OGCG: %[[RESULT_IMAG:.*]] = call float @llvm.experimental.constrained.fsub.f32(float %[[A_IMAG]], float %[[B_IMAG]], metadata !"round.tonearest", metadata !"fpexcept.strict") #2
// OGCG: %[[C_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[C_ADDR]], i32 0, i32 0
// OGCG: %[[C_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[C_ADDR]], i32 0, i32 1
// OGCG: store float %[[RESULT_REAL]], ptr %[[C_REAL_PTR]], align 4
// OGCG: store float %[[RESULT_IMAG]], ptr %[[C_IMAG_PTR]], align 4
