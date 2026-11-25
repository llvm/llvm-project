// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -ffp-contract=on -fexperimental-strict-floating-point -ffp-exception-behavior=strict -emit-cir %s -o %t-strict.cir
// RUN: FileCheck --input-file=%t-strict.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -ffp-contract=on -fexperimental-strict-floating-point -ffp-exception-behavior=strict -emit-llvm %s -o %t-strict.ll
// RUN: FileCheck --input-file=%t-strict.ll %s -check-prefixes=LLVM,SHARED
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -ffp-contract=on -fexperimental-strict-floating-point -ffp-exception-behavior=strict -emit-llvm %s -o %t-strict-ogcg.ll
// RUN: FileCheck --input-file=%t-strict-ogcg.ll %s -check-prefixes=OGCG,SHARED
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -ffp-contract=on -fexperimental-strict-floating-point -ffp-exception-behavior=strict -emit-cir -mmlir --mlir-print-ir-before=cir-canonicalize %s -o %t-strict-before.cir 2>&1 | FileCheck %s -check-prefix=CIR-BEFORE

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
// CIR: %[[RESULT:.*]] = cir.complex.fadd %[[TMP_A]], %[[TMP_B]] : !cir.complex<!cir.float> fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>
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
// CIR: %[[RESULT:.*]] = cir.complex.fsub %[[TMP_A]], %[[TMP_B]] : !cir.complex<!cir.float> fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>
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

// With CX_LIMITED_RANGE, multiplication and division expand into scalar
// arithmetic only. LoweringPrepare does the expansion, so the CIR prefix sees
// the scalar operations. Each of them carries the attribute of the complex
// operation. The NaN check, the range check and the runtime library calls of
// the other ranges are tested in complex-mul-div-fenv-access.c.
//
// ClangIR and classic codegen emit the same constrained intrinsics, so the
// SHARED lines check both. They end before the call-site attributes.
// FIXME: ClangIR does not mark the calls in a strictfp function `strictfp`.

void complex_mul() {
#pragma STDC CX_LIMITED_RANGE ON
  _Complex float a;
  _Complex float b;
  _Complex float c = a * b;
}

// CIR-BEFORE: cir.func {{.*}}@_Z11complex_mulv()
// CIR-BEFORE: %{{.*}} = cir.complex.fmul %{{.*}}, %{{.*}} range(basic) : !cir.complex<!cir.float> fenv<{{.*}}except_mode = unknown, strict_except = true>

// CIR: cir.func {{.*}}@_Z11complex_mulv()
// CIR-COUNT-6: = cir.f{{mul|sub|add}} %{{.*}}, %{{.*}} : !cir.float fenv<{{.*}}except_mode = unknown, strict_except = true>

// SHARED: define {{.*}} @_Z11complex_mulv()
// SHARED-COUNT-6: call float @llvm.experimental.constrained.f{{mul|sub|add}}.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")

void complex_div() {
#pragma STDC CX_LIMITED_RANGE ON
  _Complex float a;
  _Complex float b;
  _Complex float c = a / b;
}

// CIR-BEFORE: cir.func {{.*}}@_Z11complex_divv()
// CIR-BEFORE: %{{.*}} = cir.complex.fdiv %{{.*}}, %{{.*}} range(basic) : !cir.complex<!cir.float> fenv<{{.*}}except_mode = unknown, strict_except = true>

// CIR: cir.func {{.*}}@_Z11complex_divv()
// CIR-COUNT-11: = cir.f{{mul|add|sub|div}} %{{.*}}, %{{.*}} : !cir.float fenv<{{.*}}except_mode = unknown, strict_except = true>

// SHARED: define {{.*}} @_Z11complex_divv()
// SHARED-COUNT-11: call float @llvm.experimental.constrained.f{{mul|add|sub|div}}.f32(float %{{.*}}, float %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
