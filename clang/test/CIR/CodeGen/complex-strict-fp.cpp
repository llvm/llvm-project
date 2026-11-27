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

// LoweringPrepare expands a complex cast into scalar casts of the real and
// the imaginary part. Each of them carries the attribute of the complex cast.
// ClangIR and classic codegen emit the same constrained intrinsics for them.

void complex_cast() {
  _Complex float a;
  _Complex double b = a;
  _Complex int c = a;
  _Complex float d = c;
}

// CIR-BEFORE: cir.func {{.*}}@_Z12complex_castv()
// CIR-BEFORE: %{{.*}} = cir.cast float_complex %{{.*}} : !cir.complex<!cir.float> -> !cir.complex<!cir.double> fenv<{{.*}}except_mode = unknown, strict_except = true>
// CIR-BEFORE: %{{.*}} = cir.cast float_complex_to_int_complex %{{.*}} : !cir.complex<!cir.float> -> !cir.complex<!s32i> fenv<{{.*}}except_mode = unknown, strict_except = true>
// CIR-BEFORE: %{{.*}} = cir.cast int_complex_to_float_complex %{{.*}} : !cir.complex<!s32i> -> !cir.complex<!cir.float> fenv<{{.*}}except_mode = unknown, strict_except = true>

// CIR: cir.func {{.*}}@_Z12complex_castv()
// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!cir.complex<!cir.float>>
// CIR: %[[C_ADDR:.*]] = cir.alloca "c" {{.*}} init : !cir.ptr<!cir.complex<!s32i>>
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: %[[A_REAL:.*]] = cir.complex.real %[[TMP_A]] : !cir.complex<!cir.float> -> !cir.float
// CIR: %[[A_IMAG:.*]] = cir.complex.imag %[[TMP_A]] : !cir.complex<!cir.float> -> !cir.float
// CIR: %[[B_REAL:.*]] = cir.cast floating %[[A_REAL]] : !cir.float -> !cir.double [[FENV:fenv<.*except_mode = unknown, strict_except = true>]]
// CIR: %[[B_IMAG:.*]] = cir.cast floating %[[A_IMAG]] : !cir.float -> !cir.double [[FENV]]
// CIR: %{{.*}} = cir.complex.create %[[B_REAL]], %[[B_IMAG]] : !cir.double -> !cir.complex<!cir.double>
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: %[[A_REAL:.*]] = cir.complex.real %[[TMP_A]] : !cir.complex<!cir.float> -> !cir.float
// CIR: %[[A_IMAG:.*]] = cir.complex.imag %[[TMP_A]] : !cir.complex<!cir.float> -> !cir.float
// CIR: %[[C_REAL:.*]] = cir.cast float_to_int %[[A_REAL]] : !cir.float -> !s32i [[FENV]]
// CIR: %[[C_IMAG:.*]] = cir.cast float_to_int %[[A_IMAG]] : !cir.float -> !s32i [[FENV]]
// CIR: %{{.*}} = cir.complex.create %[[C_REAL]], %[[C_IMAG]] : !s32i -> !cir.complex<!s32i>
// CIR: %[[TMP_C:.*]] = cir.load {{.*}} %[[C_ADDR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR: %[[C_REAL:.*]] = cir.complex.real %[[TMP_C]] : !cir.complex<!s32i> -> !s32i
// CIR: %[[C_IMAG:.*]] = cir.complex.imag %[[TMP_C]] : !cir.complex<!s32i> -> !s32i
// CIR: %[[D_REAL:.*]] = cir.cast int_to_float %[[C_REAL]] : !s32i -> !cir.float [[FENV]]
// CIR: %[[D_IMAG:.*]] = cir.cast int_to_float %[[C_IMAG]] : !s32i -> !cir.float [[FENV]]
// CIR: %{{.*}} = cir.complex.create %[[D_REAL]], %[[D_IMAG]] : !cir.float -> !cir.complex<!cir.float>

// SHARED: define {{.*}} @_Z12complex_castv()
// SHARED-COUNT-2: call double @llvm.experimental.constrained.fpext.f64.f32(float %{{.*}}, metadata !"fpexcept.strict")
// SHARED-COUNT-2: call i32 @llvm.experimental.constrained.fptosi.i32.f32(float %{{.*}}, metadata !"fpexcept.strict")
// SHARED-COUNT-2: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 %{{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")

// A conversion to bool compares both parts with zero. The comparison is quiet.

void complex_to_bool() {
  _Complex float a;
  bool b = a;
}

// CIR-BEFORE: cir.func {{.*}}@_Z15complex_to_boolv()
// CIR-BEFORE: %{{.*}} = cir.cast float_complex_to_bool %{{.*}} : !cir.complex<!cir.float> -> !cir.bool fenv<{{.*}}except_mode = unknown, strict_except = true>

// CIR: cir.func {{.*}}@_Z15complex_to_boolv()
// CIR: %[[A_REAL:.*]] = cir.complex.real %{{.*}} : !cir.complex<!cir.float> -> !cir.float
// CIR: %[[A_IMAG:.*]] = cir.complex.imag %{{.*}} : !cir.complex<!cir.float> -> !cir.float
// CIR: %[[REAL_BOOL:.*]] = cir.cast float_to_bool %[[A_REAL]] : !cir.float -> !cir.bool [[FENV:fenv<.*except_mode = unknown, strict_except = true>]]
// CIR: %[[IMAG_BOOL:.*]] = cir.cast float_to_bool %[[A_IMAG]] : !cir.float -> !cir.bool [[FENV]]
// CIR: %{{.*}} = cir.select if %[[REAL_BOOL]] then %{{.*}} else %[[IMAG_BOOL]] : (!cir.bool, !cir.bool, !cir.bool) -> !cir.bool

// SHARED: define {{.*}} @_Z15complex_to_boolv()
// SHARED: %[[REAL_BOOL:.*]] = call i1 @llvm.experimental.constrained.fcmp.f32(float %{{.*}}, float 0.000000e+00, metadata !"une", metadata !"fpexcept.strict")
// SHARED: %[[IMAG_BOOL:.*]] = call i1 @llvm.experimental.constrained.fcmp.f32(float %{{.*}}, float 0.000000e+00, metadata !"une", metadata !"fpexcept.strict")
// SHARED: or i1 %[[REAL_BOOL]], %[[IMAG_BOOL]]

void float_to_bool() {
  float a;
  bool b = a;
}

// CIR: cir.func {{.*}}@_Z13float_to_boolv()
// CIR: %{{.*}} = cir.cast float_to_bool %{{.*}} : !cir.float -> !cir.bool fenv<{{.*}}except_mode = unknown, strict_except = true>

// SHARED: define {{.*}} @_Z13float_to_boolv()
// SHARED: call i1 @llvm.experimental.constrained.fcmp.f32(float %{{.*}}, float 0.000000e+00, metadata !"une", metadata !"fpexcept.strict")
