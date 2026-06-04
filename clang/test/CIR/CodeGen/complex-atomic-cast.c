// RUN: %clang_cc1 -std=c23 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c23 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c23 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

void complex_to_atomic_complex() {
  _Complex int a;
  _Atomic _Complex int b = a;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!cir.complex<!s32i>>
// CIR: %[[B_ADDR:.*]] = cir.alloca "b" {{.*}} init : !cir.ptr<!cir.complex<!s32i>>
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR: cir.store {{.*}} %[[TMP_A]], %[[B_ADDR]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>

// LLVM: %[[A_ADDR:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[B_ADDR:.*]] = alloca { i32, i32 }, i64 1, align 8
// LLVM: %[[TMP_A:.*]] = load { i32, i32 }, ptr %[[A_ADDR]], align 4
// LLVM: store { i32, i32 } %[[TMP_A]], ptr %[[B_ADDR]], align 8

// OGCG: %[[A_ADDR:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[B_ADDR:.*]] = alloca { i32, i32 }, align 8
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load i32, ptr %[[A_REAL_PTR]], align 4
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load i32, ptr %[[A_IMAG_PTR]], align 4
// OGCG: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG: store i32 %[[A_REAL]], ptr %[[B_REAL_PTR]], align 8
// OGCG: store i32 %[[A_IMAG]], ptr %[[B_IMAG_PTR]], align 4

void atomic_complex_to_complex() {
  _Atomic _Complex int a;
  _Complex int b = a;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!cir.complex<!s32i>>
// CIR: %[[B_ADDR:.*]] = cir.alloca "b" {{.*}} init : !cir.ptr<!cir.complex<!s32i>>
// CIR: %[[ATOMIC_TMP_ADDR:.*]] = cir.alloca "atomic-temp" {{.*}} : !cir.ptr<!cir.complex<!s32i>>
// CIR: %[[A_U64I:.*]] = cir.cast bitcast %[[A_ADDR]] : !cir.ptr<!cir.complex<!s32i>> -> !cir.ptr<!u64i>
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} atomic(seq_cst) %[[A_U64I]] : !cir.ptr<!u64i>, !u64i
// CIR: %[[ATOMIC_TMP_U64I:.*]] = cir.cast bitcast %[[ATOMIC_TMP_ADDR]] : !cir.ptr<!cir.complex<!s32i>> -> !cir.ptr<!u64i>
// CIR: cir.store {{.*}} %[[TMP_A]], %[[ATOMIC_TMP_U64I]] : !u64i, !cir.ptr<!u64i>
// CIR: %[[TMP_ATOMIC:.*]] = cir.load {{.*}} %[[ATOMIC_TMP_ADDR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR: cir.store {{.*}} %[[TMP_ATOMIC]], %[[B_ADDR]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>

// LLVM: %[[A_ADDR:.*]] = alloca { i32, i32 }, i64 1, align 8
// LLVM: %[[B_ADDR:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[ATOMIC_TMP_ADDR:.*]] = alloca { i32, i32 }, i64 1, align 8
// LLVM: %[[TMP_A:.*]] = load atomic i64, ptr %[[A_ADDR]] seq_cst, align 8
// LLVM: store i64 %[[TMP_A]], ptr %[[ATOMIC_TMP_ADDR]], align 8
// LLVM: %[[TMP_ATOMIC:.*]] = load { i32, i32 }, ptr %[[ATOMIC_TMP_ADDR]], align 8
// LLVM: store { i32, i32 } %[[TMP_ATOMIC]], ptr %[[B_ADDR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca { i32, i32 }, align 8
// OGCG: %[[B_ADDR:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[ATOMIC_TMP_ADDR:.*]] = alloca { i32, i32 }, align 8
// OGCG: %[[TMP_A:.*]] = load atomic i64, ptr %[[A_ADDR]] seq_cst, align 8
// OGCG: store i64 %[[TMP_A]], ptr %[[ATOMIC_TMP_ADDR]], align 8
// OGCG: %[[ATOMIC_TMP_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[ATOMIC_TMP_ADDR]], i32 0, i32 0
// OGCG: %[[ATOMIC_TMP_REAL:.*]] = load i32, ptr %[[ATOMIC_TMP_REAL_PTR]], align 8
// OGCG: %[[ATOMIC_TMP_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[ATOMIC_TMP_ADDR]], i32 0, i32 1
// OGCG: %[[ATOMIC_TMP_IMAG:.*]] = load i32, ptr %[[ATOMIC_TMP_IMAG_PTR]], align 4
// OGCG: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG: store i32 %[[ATOMIC_TMP_REAL]], ptr %[[B_REAL_PTR]], align 4
// OGCG: store i32 %[[ATOMIC_TMP_IMAG]], ptr %[[B_IMAG_PTR]], align 4

void explicit_cast_scalar_to_atomic_complex() {
  _Atomic _Complex float a = (_Atomic _Complex float)2.0f;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a", init]
// CIR: %[[CONST_2F:.*]] = cir.const #cir.fp<2.000000e+00> : !cir.float
// CIR: %[[CONST_0F:.*]] = cir.const #cir.fp<0.000000e+00> : !cir.float
// CIR: %[[COMPLEX:.*]] = cir.complex.create %[[CONST_2F]], %[[CONST_0F]] : !cir.float -> !cir.complex<!cir.float>
// CIR: cir.store {{.*}} %[[COMPLEX]], %[[A_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// LLVM: %[[A_ADDR:.*]] = alloca { float, float }, i64 1, align 8
// LLVM: store { float, float } { float 2.000000e+00, float 0.000000e+00 }, ptr %[[A_ADDR]], align 8

// OGCG: %[[A_ADDR:.*]] = alloca { float, float }, align 8
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: store float 2.000000e+00, ptr %[[A_REAL_PTR]], align 8
// OGCG: store float 0.000000e+00, ptr %[[A_IMAG_PTR]], align 4

void explicit_cast_atomic_complex_to_complex() {
  _Atomic _Complex float a = 2.0f;
  _Complex int b = (_Complex int)a;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a", init]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["b", init]
// CIR: %[[ATOMIC_TMP_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["atomic-temp"]
// CIR: %[[CONST_2F:.*]] = cir.const #cir.fp<2.000000e+00> : !cir.float
// CIR: %[[CONST_0F:.*]] = cir.const #cir.fp<0.000000e+00> : !cir.float
// CIR: %[[COMPLEX:.*]] = cir.complex.create %[[CONST_2F]], %[[CONST_0F]] : !cir.float -> !cir.complex<!cir.float>
// CIR: cir.store {{.*}} %[[COMPLEX]], %[[A_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>
// CIR: %[[A_U64I:.*]] = cir.cast bitcast %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>> -> !cir.ptr<!u64i>
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} atomic(seq_cst) %[[A_U64I]] : !cir.ptr<!u64i>, !u64i
// CIR: %[[ATOMIC_TMP_U64I:.*]] = cir.cast bitcast %[[ATOMIC_TMP_ADDR]] : !cir.ptr<!cir.complex<!cir.float>> -> !cir.ptr<!u64i>
// CIR: cir.store {{.*}} %[[TMP_A]], %[[ATOMIC_TMP_U64I]] : !u64i, !cir.ptr<!u64i>
// CIR: %[[TMP_ATOMIC:.*]] = cir.load {{.*}} %[[ATOMIC_TMP_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: %[[ATOMIC_TMP_REAL:.*]] = cir.complex.real %[[TMP_ATOMIC]] : !cir.complex<!cir.float> -> !cir.float
// CIR: %[[ATOMIC_TMP_IMAG:.*]] = cir.complex.imag %[[TMP_ATOMIC]] : !cir.complex<!cir.float> -> !cir.floa
// CIR: %[[ATOMIC_TMP_REAL_I32:.*]] = cir.cast float_to_int %[[ATOMIC_TMP_REAL]] : !cir.float -> !s32i
// CIR: %[[ATOMIC_TMP_IMAG_I32:.*]] = cir.cast float_to_int %[[ATOMIC_TMP_IMAG]] : !cir.float -> !s32i
// CIR: %[[RESULT:.*]] = cir.complex.create %[[ATOMIC_TMP_REAL_I32]], %[[ATOMIC_TMP_IMAG_I32]] : !s32i -> !cir.complex<!s32i>
// CIR: cir.store {{.*}} %[[RESULT]], %[[B_ADDR]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>

// LLVM: %[[A_ADDR:.*]] = alloca { float, float }, i64 1, align 8
// LLVM: %[[B_ADDR:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[ATOMIC_TMP_ADDR:.*]] = alloca { float, float }, i64 1, align 8
// LLVM: store { float, float } { float 2.000000e+00, float 0.000000e+00 }, ptr %[[A_ADDR]], align 8
// LLVM: %[[TMP_A:.*]] = load atomic i64, ptr %[[A_ADDR]] seq_cst, align 8
// LLVM: store i64 %[[TMP_A]], ptr %[[ATOMIC_TMP_ADDR]], align 8
// LLVM: %[[TMP_ATOMIC:.*]] = load { float, float }, ptr %[[ATOMIC_TMP_ADDR]], align 8
// LLVM: %[[ATOMIC_TMP_REAL:.*]] = extractvalue { float, float } %[[TMP_ATOMIC]], 0
// LLVM: %[[ATOMIC_TMP_IMAG:.*]] = extractvalue { float, float } %[[TMP_ATOMIC]], 1
// LLVM: %[[ATOMIC_TMP_REAL_I32:.*]] = fptosi float %[[ATOMIC_TMP_REAL]] to i32
// LLVM: %[[ATOMIC_TMP_IMAG_I32:.*]] = fptosi float %[[ATOMIC_TMP_IMAG]] to i32
// LLVM: %[[TMP_RESULT:.*]] = insertvalue { i32, i32 } {{.*}}, i32 %[[ATOMIC_TMP_REAL_I32]], 0
// LLVM: %[[RESULT:.*]] = insertvalue { i32, i32 } %[[TMP_RESULT]], i32 %[[ATOMIC_TMP_IMAG_I32]], 1
// LLVM: store { i32, i32 } %[[RESULT]], ptr %[[B_ADDR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca { float, float }, align 8
// OGCG: %[[B_ADDR:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[ATOMIC_TMP_ADDR:.*]] = alloca { float, float }, align 8
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: store float 2.000000e+00, ptr %[[A_REAL_PTR]], align 8
// OGCG: store float 0.000000e+00, ptr %[[A_IMAG_PTR]], align 4
// OGCG: %[[TMP_A:.*]] = load atomic i64, ptr %[[A_ADDR]] seq_cst, align 8
// OGCG: store i64 %[[TMP_A]], ptr %[[ATOMIC_TMP_ADDR]], align 8
// OGCG: %[[ATOMIC_TMP_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[ATOMIC_TMP_ADDR]], i32 0, i32 0
// OGCG: %[[ATOMIC_TMP_REAL:.*]] = load float, ptr %[[ATOMIC_TMP_REAL_PTR]], align 8
// OGCG: %[[ATOMIC_TMP_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[ATOMIC_TMP_ADDR]], i32 0, i32 1
// OGCG: %[[ATOMIC_TMP_IMAG:.*]] = load float, ptr %[[ATOMIC_TMP_IMAG_PTR]], align 4
// OGCG: %[[RESULT_REAL:.*]] = fptosi float %[[ATOMIC_TMP_REAL]] to i32
// OGCG: %[[RESULT_IMAG:.*]] = fptosi float %[[ATOMIC_TMP_IMAG]] to i32
// OGCG: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG: store i32 %[[RESULT_REAL]], ptr %[[B_REAL_PTR]], align 4
// OGCG: store i32 %[[RESULT_IMAG]], ptr %[[B_IMAG_PTR]], align 4

void explicit_cast_atomic_complex_to_atomic_complex() {
  _Atomic _Complex float a = 2.0f;
  _Atomic _Complex int b = (_Atomic _Complex int)a;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a", init]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["b", init]
// CIR: %[[ATOMIC_TMP_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["atomic-temp"]
// CIR: %[[CONST_2F:.*]] = cir.const #cir.fp<2.000000e+00> : !cir.float
// CIR: %[[CONST_0F:.*]] = cir.const #cir.fp<0.000000e+00> : !cir.float
// CIR: %[[COMPLEX:.*]] = cir.complex.create %[[CONST_2F]], %[[CONST_0F]] : !cir.float -> !cir.complex<!cir.float>
// CIR: cir.store {{.*}} %[[COMPLEX]], %[[A_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>
// CIR: %[[A_U64I:.*]] = cir.cast bitcast %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>> -> !cir.ptr<!u64i>
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} atomic(seq_cst) %[[A_U64I]] : !cir.ptr<!u64i>, !u64i
// CIR: %[[ATOMIC_TMP_U64I:.*]] = cir.cast bitcast %[[ATOMIC_TMP_ADDR]] : !cir.ptr<!cir.complex<!cir.float>> -> !cir.ptr<!u64i>
// CIR: cir.store {{.*}} %[[TMP_A]], %[[ATOMIC_TMP_U64I]] : !u64i, !cir.ptr<!u64i>
// CIR: %[[TMP_ATOMIC:.*]] = cir.load {{.*}} %[[ATOMIC_TMP_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: %[[ATOMIC_TMP_REAL:.*]] = cir.complex.real %[[TMP_ATOMIC]] : !cir.complex<!cir.float> -> !cir.float
// CIR: %[[ATOMIC_TMP_IMAG:.*]] = cir.complex.imag %[[TMP_ATOMIC]] : !cir.complex<!cir.float> -> !cir.float
// CIR: %[[ATOMIC_TMP_REAL_I32:.*]] = cir.cast float_to_int %[[ATOMIC_TMP_REAL]] : !cir.float -> !s32i
// CIR: %[[ATOMIC_TMP_IMAG_I32:.*]] = cir.cast float_to_int %[[ATOMIC_TMP_IMAG]] : !cir.float -> !s32i
// CIR: %[[RESULT:.*]] = cir.complex.create %[[ATOMIC_TMP_REAL_I32]], %[[ATOMIC_TMP_IMAG_I32]] : !s32i -> !cir.complex<!s32i>
// CIR: cir.store {{.*}} %[[RESULT]], %[[B_ADDR]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>

// LLVM: %[[A_ADDR:.*]] = alloca { float, float }, i64 1, align 8
// LLVM: %[[B_ADDR:.*]] = alloca { i32, i32 }, i64 1, align 8
// LLVM: %[[ATOMIC_TMP_ADDR:.*]] = alloca { float, float }, i64 1, align 8
// LLVM: store { float, float } { float 2.000000e+00, float 0.000000e+00 }, ptr %[[A_ADDR]], align 8
// LLVM: %[[TMP_A:.*]] = load atomic i64, ptr %[[A_ADDR]] seq_cst, align 8
// LLVM: store i64 %[[TMP_A]], ptr %[[ATOMIC_TMP_ADDR]], align 8
// LLVM: %[[TMP_ATOMIC:.*]] = load { float, float }, ptr %[[ATOMIC_TMP_ADDR]], align 8
// LLVM: %[[ATOMIC_TMP_REAL:.*]] = extractvalue { float, float } %[[TMP_ATOMIC]], 0
// LLVM: %[[ATOMIC_TMP_IMAG:.*]] = extractvalue { float, float } %[[TMP_ATOMIC]], 1
// LLVM: %[[ATOMIC_TMP_REAL_I32:.*]] = fptosi float %[[ATOMIC_TMP_REAL]] to i32
// LLVM: %[[ATOMIC_TMP_IMAG_I32:.*]] = fptosi float %[[ATOMIC_TMP_IMAG]] to i32
// LLVM: %[[TMP_RESULT:.*]] = insertvalue { i32, i32 } {{.*}}, i32 %[[ATOMIC_TMP_REAL_I32]], 0
// LLVM: %[[RESULT:.*]] = insertvalue { i32, i32 } %[[TMP_RESULT]], i32 %[[ATOMIC_TMP_IMAG_I32]], 1
// LLVM: store { i32, i32 } %[[RESULT]], ptr %[[B_ADDR]], align 8

// OGCG: %[[A_ADDR:.*]] = alloca { float, float }, align 8
// OGCG: %[[B_ADDR:.*]] = alloca { i32, i32 }, align 8
// OGCG: %[[ATOMIC_TMP_ADDR:.*]] = alloca { float, float }, align 8
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: store float 2.000000e+00, ptr %[[A_REAL_PTR]], align 8
// OGCG: store float 0.000000e+00, ptr %[[A_IMAG_PTR]], align 4
// OGCG: %[[TMP_A:.*]] = load atomic i64, ptr %[[A_ADDR]] seq_cst, align 8
// OGCG: store i64 %[[TMP_A]], ptr %[[ATOMIC_TMP_ADDR]], align 8
// OGCG: %[[ATOMIC_TMP_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[ATOMIC_TMP_ADDR]], i32 0, i32 0
// OGCG: %[[ATOMIC_TMP_REAL:.*]] = load float, ptr %[[ATOMIC_TMP_REAL_PTR]], align 8
// OGCG: %[[ATOMIC_TMP_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[ATOMIC_TMP_ADDR]], i32 0, i32 1
// OGCG: %[[ATOMIC_TMP_IMAG:.*]] = load float, ptr %[[ATOMIC_TMP_IMAG_PTR]], align 4
// OGCG: %[[RESULT_REAL:.*]] = fptosi float %[[ATOMIC_TMP_REAL]] to i32
// OGCG: %[[RESULT_IMAG:.*]] = fptosi float %[[ATOMIC_TMP_IMAG]] to i32
// OGCG: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG: store i32 %[[RESULT_REAL]], ptr %[[B_REAL_PTR]], align 8
// OGCG: store i32 %[[RESULT_IMAG]], ptr %[[B_IMAG_PTR]], align 4
