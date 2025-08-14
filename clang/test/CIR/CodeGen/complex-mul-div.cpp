// complex-range basic
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -complex-range=basic -Wno-unused-value -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-canonicalize -o %t.cir %s 2>&1 | FileCheck --check-prefix=CIR-BEFORE-BASIC %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -complex-range=basic -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefixes=CIR-AFTER-INT,CIR-AFTER-MUL-COMBINED,CIR-COMBINED
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -complex-range=basic -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefixes=LLVM-INT,LLVM-MUL-COMBINED,LLVM-COMBINED
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -complex-range=basic -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefixes=OGCG-INT,OGCG-MUL-COMBINED,OGCG-COMBINED

// complex-range improved
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -complex-range=improved -Wno-unused-value -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-canonicalize -o %t.cir %s 2>&1 | FileCheck --check-prefix=CIR-BEFORE-IMPROVED %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -complex-range=improved -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefixes=CIR-AFTER-INT,CIR-AFTER-MUL-COMBINED,CIR-COMBINED
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -complex-range=improved -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefixes=LLVM-INT,LLVM-MUL-COMBINED,LLVM-COMBINED
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -complex-range=improved -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefixes=OGCG-INT,OGCG-MUL-COMBINED,OGCG-COMBINED

// complex-range promoted
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -complex-range=promoted -Wno-unused-value -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-canonicalize -o %t.cir %s 2>&1 | FileCheck --check-prefix=CIR-BEFORE-PROMOTED %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -complex-range=promoted -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefixes=CIR-AFTER-INT,CIR-AFTER-MUL-COMBINED,CIR-COMBINED
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -complex-range=promoted -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefixes=LLVM-INT,LLVM-MUL-COMBINED,LLVM-COMBINED
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -complex-range=promoted -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefixes=OGCG-INT,OGCG-MUL-COMBINED,OGCG-COMBINED

// complex-range full
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -complex-range=full -Wno-unused-value -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-canonicalize -o %t.cir %s 2>&1 | FileCheck --check-prefix=CIR-BEFORE-FULL %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -complex-range=full -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefixes=CIR-AFTER-FULL,CIR-AFTER-INT,CIR-COMBINED
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -complex-range=full -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefixes=LLVM-FULL,LLVM-INT,LLVM-COMBINED
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -complex-range=full -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefixes=OGCG-FULL,OGCG-INT,OGCG-COMBINED

void foo() {
  float _Complex a;
  float _Complex b;
  float _Complex c = a * b;
}

// CIR-BEFORE-BASIC: %{{.*}} = cir.complex.mul {{.*}}, {{.*}} range(basic) : !cir.complex<!cir.float>

// CIR-BEFORE-IMPROVED: %{{.*}} = cir.complex.mul {{.*}}, {{.*}} range(improved) : !cir.complex<!cir.float>

// CIR-BEFORE-PROMOTED: %{{.*}} = cir.complex.mul {{.*}}, {{.*}} range(promoted) : !cir.complex<!cir.float>

// CIR-AFTER-MUL-COMBINED: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR-AFTER-MUL-COMBINED: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["b"]
// CIR-AFTER-MUL-COMBINED: %[[C_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["c", init]
// CIR-AFTER-MUL-COMBINED: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR-AFTER-MUL-COMBINED: %[[TMP_B:.*]] = cir.load{{.*}} %[[B_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR-AFTER-MUL-COMBINED: %[[A_REAL:.*]] = cir.complex.real %[[TMP_A]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER-MUL-COMBINED: %[[A_IMAG:.*]] = cir.complex.imag %[[TMP_A]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER-MUL-COMBINED: %[[B_REAL:.*]] = cir.complex.real %[[TMP_B]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER-MUL-COMBINED: %[[B_IMAG:.*]] = cir.complex.imag %[[TMP_B]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER-MUL-COMBINED: %[[MUL_AR_BR:.*]] = cir.binop(mul, %[[A_REAL]], %[[B_REAL]]) : !cir.float
// CIR-AFTER-MUL-COMBINED: %[[MUL_AI_BI:.*]] = cir.binop(mul, %[[A_IMAG]], %[[B_IMAG]]) : !cir.float
// CIR-AFTER-MUL-COMBINED: %[[MUL_AR_BI:.*]] = cir.binop(mul, %[[A_REAL]], %[[B_IMAG]]) : !cir.float
// CIR-AFTER-MUL-COMBINED: %[[MUL_AI_BR:.*]] = cir.binop(mul, %[[A_IMAG]], %[[B_REAL]]) : !cir.float
// CIR-AFTER-MUL-COMBINED: %[[C_REAL:.*]] = cir.binop(sub, %[[MUL_AR_BR]], %[[MUL_AI_BI]]) : !cir.float
// CIR-AFTER-MUL-COMBINED: %[[C_IMAG:.*]] = cir.binop(add, %[[MUL_AR_BI]], %[[MUL_AI_BR]]) : !cir.float
// CIR-AFTER-MUL-COMBINED: %[[RESULT:.*]] = cir.complex.create %[[C_REAL]], %[[C_IMAG]] : !cir.float -> !cir.complex<!cir.float>
// CIR-AFTER-MUL-COMBINED: cir.store{{.*}} %[[RESULT]], %[[C_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// LLVM-MUL-COMBINED: %[[A_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM-MUL-COMBINED: %[[B_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM-MUL-COMBINED: %[[C_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM-MUL-COMBINED: %[[TMP_A:.*]] = load { float, float }, ptr %[[A_ADDR]], align 4
// LLVM-MUL-COMBINED: %[[TMP_B:.*]] = load { float, float }, ptr %[[B_ADDR]], align 4
// LLVM-MUL-COMBINED: %[[A_REAL:.*]] = extractvalue { float, float } %[[TMP_A]], 0
// LLVM-MUL-COMBINED: %[[A_IMAG:.*]] = extractvalue { float, float } %[[TMP_A]], 1
// LLVM-MUL-COMBINED: %[[B_REAL:.*]] = extractvalue { float, float } %[[TMP_B]], 0
// LLVM-MUL-COMBINED: %[[B_IMAG:.*]] = extractvalue { float, float } %[[TMP_B]], 1
// LLVM-MUL-COMBINED: %[[MUL_AR_BR:.*]] = fmul float %[[A_REAL]], %[[B_REAL]]
// LLVM-MUL-COMBINED: %[[MUL_AI_BI:.*]] = fmul float %[[A_IMAG]], %[[B_IMAG]]
// LLVM-MUL-COMBINED: %[[MUL_AR_BI:.*]] = fmul float %[[A_REAL]], %[[B_IMAG]]
// LLVM-MUL-COMBINED: %[[MUL_AI_BR:.*]] = fmul float %[[A_IMAG]], %[[B_REAL]]
// LLVM-MUL-COMBINED: %[[C_REAL:.*]] = fsub float %[[MUL_AR_BR]], %[[MUL_AI_BI]]
// LLVM-MUL-COMBINED: %[[C_IMAG:.*]] = fadd float %[[MUL_AR_BI]], %[[MUL_AI_BR]]
// LLVM-MUL-COMBINED: %[[MUL_A_B:.*]] = insertvalue { float, float } {{.*}}, float %[[C_REAL]], 0
// LLVM-MUL-COMBINED: %[[RESULT:.*]] = insertvalue { float, float } %[[MUL_A_B]], float %[[C_IMAG]], 1
// LLVM-MUL-COMBINED: store { float, float } %[[RESULT]], ptr %[[C_ADDR]], align 4

// OGCG-MUL-COMBINED: %[[A_ADDR:.*]] = alloca { float, float }, align 4
// OGCG-MUL-COMBINED: %[[B_ADDR:.*]] = alloca { float, float }, align 4
// OGCG-MUL-COMBINED: %[[C_ADDR:.*]] = alloca { float, float }, align 4
// OGCG-MUL-COMBINED: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG-MUL-COMBINED: %[[A_REAL:.*]] = load float, ptr %[[A_REAL_PTR]], align 4
// OGCG-MUL-COMBINED: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG-MUL-COMBINED: %[[A_IMAG:.*]] = load float, ptr %[[A_IMAG_PTR]], align 4
// OGCG-MUL-COMBINED: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG-MUL-COMBINED: %[[B_REAL:.*]] = load float, ptr %[[B_REAL_PTR]], align 4
// OGCG-MUL-COMBINED: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG-MUL-COMBINED: %[[B_IMAG:.*]] = load float, ptr %[[B_IMAG_PTR]], align 4
// OGCG-MUL-COMBINED: %[[MUL_AR_BR:.*]] = fmul float %[[A_REAL]], %[[B_REAL]]
// OGCG-MUL-COMBINED: %[[MUL_AI_BI:.*]] = fmul float %[[A_IMAG]], %[[B_IMAG]]
// OGCG-MUL-COMBINED: %[[MUL_AR_BI:.*]] = fmul float %[[A_REAL]], %[[B_IMAG]]
// OGCG-MUL-COMBINED: %[[MUL_AI_BR:.*]] = fmul float %[[A_IMAG]], %[[B_REAL]]
// OGCG-MUL-COMBINED: %[[C_REAL:.*]] = fsub float %[[MUL_AR_BR]], %[[MUL_AI_BI]]
// OGCG-MUL-COMBINED: %[[C_IMAG:.*]] = fadd float %[[MUL_AR_BI]], %[[MUL_AI_BR]]
// OGCG-MUL-COMBINED: %[[C_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[C_ADDR]], i32 0, i32 0
// OGCG-MUL-COMBINED: %[[C_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[C_ADDR]], i32 0, i32 1
// OGCG-MUL-COMBINED: store float %[[C_REAL]], ptr %[[C_REAL_PTR]], align 4
// OGCG-MUL-COMBINED: store float %[[C_IMAG]], ptr %[[C_IMAG_PTR]], align 4

// CIR-BEFORE-FULL: %{{.*}} = cir.complex.mul {{.*}}, {{.*}} range(full) : !cir.complex<!cir.float>

// CIR-AFTER-FULL: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR-AFTER-FULL: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["b"]
// CIR-AFTER-FULL: %[[C_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["c", init]
// CIR-AFTER-FULL: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR-AFTER-FULL: %[[TMP_B:.*]] = cir.load{{.*}} %[[B_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR-AFTER-FULL: %[[A_REAL:.*]] = cir.complex.real %[[TMP_A]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER-FULL: %[[A_IMAG:.*]] = cir.complex.imag %[[TMP_A]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER-FULL: %[[B_REAL:.*]] = cir.complex.real %[[TMP_B]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER-FULL: %[[B_IMAG:.*]] = cir.complex.imag %[[TMP_B]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER-FULL: %[[MUL_AR_BR:.*]] = cir.binop(mul, %[[A_REAL]], %[[B_REAL]]) : !cir.float
// CIR-AFTER-FULL: %[[MUL_AI_BI:.*]] = cir.binop(mul, %[[A_IMAG]], %[[B_IMAG]]) : !cir.float
// CIR-AFTER-FULL: %[[MUL_AR_BI:.*]] = cir.binop(mul, %[[A_REAL]], %[[B_IMAG]]) : !cir.float
// CIR-AFTER-FULL: %[[MUL_AI_BR:.*]] = cir.binop(mul, %[[A_IMAG]], %[[B_REAL]]) : !cir.float
// CIR-AFTER-FULL: %[[C_REAL:.*]] = cir.binop(sub, %[[MUL_AR_BR]], %[[MUL_AI_BI]]) : !cir.float
// CIR-AFTER-FULL: %[[C_IMAG:.*]] = cir.binop(add, %[[MUL_AR_BI]], %[[MUL_AI_BR]]) : !cir.float
// CIR-AFTER-FULL: %[[COMPLEX:.*]] = cir.complex.create %[[C_REAL]], %[[C_IMAG]] : !cir.float -> !cir.complex<!cir.float>
// CIR-AFTER-FULL: %[[IS_C_REAL_NAN:.*]] = cir.cmp(ne, %[[C_REAL]], %[[C_REAL]]) : !cir.float, !cir.bool
// CIR-AFTER-FULL: %[[IS_C_IMAG_NAN:.*]] = cir.cmp(ne, %[[C_IMAG]], %[[C_IMAG]]) : !cir.float, !cir.bool
// CIR-AFTER-FULL: %[[CONST_FALSE:.*]] = cir.const #false
// CIR-AFTER-FULL: %[[SELECT_CONDITION:.*]] = cir.select if %[[IS_C_REAL_NAN]] then %[[IS_C_IMAG_NAN]] else %[[CONST_FALSE]] : (!cir.bool, !cir.bool, !cir.bool) -> !cir.bool
// CIR-AFTER-FULL: %[[RESULT:.*]] = cir.ternary(%[[SELECT_CONDITION]], true {
// CIR-AFTER-FULL:   %[[LIBC_COMPLEX:.*]] = cir.call @__mulsc3(%[[A_REAL]], %[[A_IMAG]], %[[B_REAL]], %[[B_IMAG]]) : (!cir.float, !cir.float, !cir.float, !cir.float) -> !cir.complex<!cir.float>
// CIR-AFTER-FULL:   cir.yield %[[LIBC_COMPLEX]] : !cir.complex<!cir.float>
// CIR-AFTER-FULL: }, false {
// CIR-AFTER-FULL:   cir.yield %[[COMPLEX]] : !cir.complex<!cir.float>
// CIR-AFTER-FULL: }) : (!cir.bool) -> !cir.complex<!cir.float>
// CIR-AFTER-FULL: cir.store{{.*}} %[[RESULT]], %[[C_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// LLVM-FULL: %[[A_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM-FULL: %[[B_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM-FULL: %[[C_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM-FULL: %[[TMP_A:.*]] = load { float, float }, ptr %[[A_ADDR]], align 4
// LLVM-FULL: %[[TMP_B:.*]] = load { float, float }, ptr %[[B_ADDR]], align 4
// LLVM-FULL: %[[A_REAL:.*]] = extractvalue { float, float } %[[TMP_A]], 0
// LLVM-FULL: %[[A_IMAG:.*]] = extractvalue { float, float } %[[TMP_A]], 1
// LLVM-FULL: %[[B_REAL:.*]] = extractvalue { float, float } %[[TMP_B]], 0
// LLVM-FULL: %[[B_IMAG:.*]] = extractvalue { float, float } %[[TMP_B]], 1
// LLVM-FULL: %[[MUL_AR_BR:.*]] = fmul float %[[A_REAL]], %[[B_REAL]]
// LLVM-FULL: %[[MUL_AI_BI:.*]] = fmul float %[[A_IMAG]], %[[B_IMAG]]
// LLVM-FULL: %[[MUL_AR_BI:.*]] = fmul float %[[A_REAL]], %[[B_IMAG]]
// LLVM-FULL: %[[MUL_AI_BR:.*]] = fmul float %[[A_IMAG]], %[[B_REAL]]
// LLVM-FULL: %[[C_REAL:.*]] = fsub float %[[MUL_AR_BR]], %[[MUL_AI_BI]]
// LLVM-FULL: %[[C_IMAG:.*]] = fadd float %[[MUL_AR_BI]], %[[MUL_AI_BR]]
// LLVM-FULL: %[[MUL_A_B:.*]] = insertvalue { float, float } {{.*}}, float %[[C_REAL]], 0
// LLVM-FULL: %[[COMPLEX:.*]] = insertvalue { float, float } %[[MUL_A_B]], float %[[C_IMAG]], 1
// LLVM-FULL: %[[IS_C_REAL_NAN:.*]] = fcmp une float %[[C_REAL]], %[[C_REAL]]
// LLVM-FULL: %[[IS_C_IMAG_NAN:.*]] = fcmp une float %[[C_IMAG]], %[[C_IMAG]]
// LLVM-FULL: %[[SELECT_CONDITION:.*]] = and i1 %[[IS_C_REAL_NAN]], %[[IS_C_IMAG_NAN]]
// LLVM-FULL: br i1 %[[SELECT_CONDITION]], label %[[THEN_LABEL:.*]], label %[[ELSE_LABEL:.*]]
// LLVM-FULL: [[THEN_LABEL]]:
// LLVM-FULL:  %[[LIBC_COMPLEX:.*]] = call { float, float } @__mulsc3(float %[[A_REAL]], float %[[A_IMAG]], float %[[B_REAL]], float %[[B_IMAG]])
// LLVM-FULL:  br label %[[PHI_BRANCH:.*]]
// LLVM-FULL: [[ELSE_LABEL]]:
// LLVM-FULL:  br label %[[PHI_BRANCH:]]
// LLVM-FULL: [[PHI_BRANCH:]]:
// LLVM-FULL:  %[[RESULT:.*]] = phi { float, float } [ %[[COMPLEX]], %[[ELSE_LABEL]] ], [ %[[LIBC_COMPLEX]], %[[THEN_LABEL]] ]
// LLVM-FULL:  br label %[[END_LABEL:.*]]
// LLVM-FULL: [[END_LABEL]]:
// LLVM-FULL:  store { float, float } %[[RESULT]], ptr %[[C_ADDR]], align 4

// OGCG-FULL: %[[A_ADDR:.*]] = alloca { float, float }, align 4
// OGCG-FULL: %[[B_ADDR:.*]] = alloca { float, float }, align 4
// OGCG-FULL: %[[C_ADDR:.*]] = alloca { float, float }, align 4
// OGCG-FULL: %[[COMPLEX_CALL_ADDR:.*]] = alloca { float, float }, align 4
// OGCG-FULL: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG-FULL: %[[A_REAL:.*]] = load float, ptr %[[A_REAL_PTR]], align 4
// OGCG-FULL: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG-FULL: %[[A_IMAG:.*]] = load float, ptr %[[A_IMAG_PTR]], align 4
// OGCG-FULL: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG-FULL: %[[B_REAL:.*]] = load float, ptr %[[B_REAL_PTR]], align 4
// OGCG-FULL: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG-FULL: %[[B_IMAG:.*]] = load float, ptr %[[B_IMAG_PTR]], align 4
// OGCG-FULL: %[[MUL_AR_BR:.*]] = fmul float %[[A_REAL]], %[[B_REAL]]
// OGCG-FULL: %[[MUL_AI_BI:.*]] = fmul float %[[A_IMAG]], %[[B_IMAG]]
// OGCG-FULL: %[[MUL_AR_BI:.*]] = fmul float %[[A_REAL]], %[[B_IMAG]]
// OGCG-FULL: %[[MUL_AI_BR:.*]] = fmul float %[[A_IMAG]], %[[B_REAL]]
// OGCG-FULL: %[[C_REAL:.*]] = fsub float %[[MUL_AR_BR]], %[[MUL_AI_BI]]
// OGCG-FULL: %[[C_IMAG:.*]] = fadd float %[[MUL_AR_BI]], %[[MUL_AI_BR]]
// OGCG-FULL: %[[IS_C_REAL_NAN:.*]] = fcmp uno float %[[C_REAL]], %[[C_REAL]]
// OGCG-FULL: br i1 %[[IS_C_REAL_NAN]], label %[[COMPLEX_IS_IMAG_NAN:.*]], label %[[END_LABEL:.*]], !prof !2
// OGCG-FULL: [[COMPLEX_IS_IMAG_NAN]]:
// OGCG-FULL:  %[[IS_C_IMAG_NAN:.*]] = fcmp uno float %[[C_IMAG]], %[[C_IMAG]]
// OGCG-FULL:  br i1 %[[IS_C_IMAG_NAN]], label %[[COMPLEX_LIB_CALL:.*]], label %[[END_LABEL]], !prof !2
// OGCG-FULL: [[COMPLEX_LIB_CALL]]:
// OGCG-FULL:  %[[CALL_RESULT:.*]] = call noundef <2 x float> @__mulsc3(float noundef %[[A_REAL]], float noundef %[[A_IMAG]], float noundef %[[B_REAL]], float noundef %[[B_IMAG]])
// OGCG-FULL:  store <2 x float> %[[CALL_RESULT]], ptr %[[COMPLEX_CALL_ADDR]], align 4
// OGCG-FULL:  %[[COMPLEX_CALL_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[COMPLEX_CALL_ADDR]], i32 0, i32 0
// OGCG-FULL:  %[[COMPLEX_CALL_REAL:.*]] = load float, ptr %[[COMPLEX_CALL_REAL_PTR]], align 4
// OGCG-FULL:  %[[COMPLEX_CALL_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[COMPLEX_CALL_ADDR]], i32 0, i32 1
// OGCG-FULL:  %[[COMPLEX_CALL_IMAG:.*]] = load float, ptr %[[COMPLEX_CALL_IMAG_PTR]], align 4
// OGCG-FULL:  br label %[[END_LABEL]]
// OGCG-FULL: [[END_LABEL]]:
// OGCG-FULL:  %[[FINAL_REAL:.*]] = phi float [ %[[C_REAL]], %[[ENTRY:.*]] ], [ %[[C_REAL]], %[[COMPLEX_IS_IMAG_NAN]] ], [ %[[COMPLEX_CALL_REAL]], %[[COMPLEX_LIB_CALL]] ]
// OGCG-FULL:  %[[FINAL_IMAG:.*]] = phi float [ %[[C_IMAG]], %[[ENTRY]] ], [ %[[C_IMAG]], %[[COMPLEX_IS_IMAG_NAN]] ], [ %[[COMPLEX_CALL_IMAG]], %[[COMPLEX_LIB_CALL]] ]
// OGCG-FULL:  %[[C_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[C_ADDR]], i32 0, i32 0
// OGCG-FULL:  %[[C_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[C_ADDR]], i32 0, i32 1
// OGCG-FULL:  store float %[[FINAL_REAL]], ptr %[[C_REAL_PTR]], align 4
// OGCG-FULL:  store float %[[FINAL_IMAG]], ptr %[[C_IMAG_PTR]], align 4

void foo1() {
  int _Complex a;
  int _Complex b;
  int _Complex c = a * b;
}

// CIR-BEFORE-BASIC: %{{.*}} = cir.complex.mul {{.*}}, {{.*}} range(basic) : !cir.complex<!s32i>

// CIR-BEFORE-IMPROVED: %{{.*}} = cir.complex.mul {{.*}}, {{.*}} range(improved) : !cir.complex<!s32i>

// CIR-BEFORE-PROMOTED: %{{.*}} = cir.complex.mul {{.*}}, {{.*}} range(promoted) : !cir.complex<!s32i>

// CIR-BEFORE-FULL: %{{.*}} = cir.complex.mul {{.*}}, {{.*}} range(full) : !cir.complex<!s32i>

// CIR-AFTER-INT: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["a"]
// CIR-AFTER-INT: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["b"]
// CIR-AFTER-INT: %[[C_ADDR:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["c", init]
// CIR-AFTER-INT: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR-AFTER-INT: %[[TMP_B:.*]] = cir.load{{.*}} %[[B_ADDR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR-AFTER-INT: %[[A_REAL:.*]] = cir.complex.real %[[TMP_A]] : !cir.complex<!s32i> -> !s32i
// CIR-AFTER-INT: %[[A_IMAG:.*]] = cir.complex.imag %[[TMP_A]] : !cir.complex<!s32i> -> !s32i
// CIR-AFTER-INT: %[[B_REAL:.*]] = cir.complex.real %[[TMP_B]] : !cir.complex<!s32i> -> !s32i
// CIR-AFTER-INT: %[[B_IMAG:.*]] = cir.complex.imag %[[TMP_B]] : !cir.complex<!s32i> -> !s32i
// CIR-AFTER-INT: %[[MUL_AR_BR:.*]] = cir.binop(mul, %[[A_REAL]], %[[B_REAL]]) : !s32i
// CIR-AFTER-INT: %[[MUL_AI_BI:.*]] = cir.binop(mul, %[[A_IMAG]], %[[B_IMAG]]) : !s32i
// CIR-AFTER-INT: %[[MUL_AR_BI:.*]] = cir.binop(mul, %[[A_REAL]], %[[B_IMAG]]) : !s32i
// CIR-AFTER-INT: %[[MUL_AI_BR:.*]] = cir.binop(mul, %[[A_IMAG]], %[[B_REAL]]) : !s32i
// CIR-AFTER-INT: %[[C_REAL:.*]] = cir.binop(sub, %[[MUL_AR_BR]], %[[MUL_AI_BI]]) : !s32i
// CIR-AFTER-INT: %[[C_IMAG:.*]] = cir.binop(add, %[[MUL_AR_BI]], %[[MUL_AI_BR]]) : !s32i
// CIR-AFTER-INT: %[[RESULT:.*]] = cir.complex.create %[[C_REAL]], %[[C_IMAG]] : !s32i -> !cir.complex<!s32i>
// CIR-AFTER-INT: cir.store{{.*}} %[[RESULT]], %[[C_ADDR]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>

// LLVM-INT: %[[A_ADDR:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM-INT: %[[B_ADDR:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM-INT: %[[C_ADDR:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM-INT: %[[TMP_A:.*]] = load { i32, i32 }, ptr %[[A_ADDR]], align 4
// LLVM-INT: %[[TMP_B:.*]] = load { i32, i32 }, ptr %[[B_ADDR]], align 4
// LLVM-INT: %[[A_REAL:.*]] = extractvalue { i32, i32 } %[[TMP_A]], 0
// LLVM-INT: %[[A_IMAG:.*]] = extractvalue { i32, i32 } %[[TMP_A]], 1
// LLVM-INT: %[[B_REAL:.*]] = extractvalue { i32, i32 } %[[TMP_B]], 0
// LLVM-INT: %[[B_IMAG:.*]] = extractvalue { i32, i32 } %[[TMP_B]], 1
// LLVM-INT: %[[MUL_AR_BR:.*]] = mul i32 %[[A_REAL]], %[[B_REAL]]
// LLVM-INT: %[[MUL_AI_BI:.*]] = mul i32 %[[A_IMAG]], %[[B_IMAG]]
// LLVM-INT: %[[MUL_AR_BI:.*]] = mul i32 %[[A_REAL]], %[[B_IMAG]]
// LLVM-INT: %[[MUL_AI_BR:.*]] = mul i32 %[[A_IMAG]], %[[B_REAL]]
// LLVM-INT: %[[C_REAL:.*]] = sub i32 %[[MUL_AR_BR]], %[[MUL_AI_BI]]
// LLVM-INT: %[[C_IMAG:.*]] = add i32 %[[MUL_AR_BI]], %[[MUL_AI_BR]]
// LLVM-INT: %[[MUL_A_B:.*]] = insertvalue { i32, i32 } {{.*}}, i32 %[[C_REAL]], 0
// LLVM-INT: %[[RESULT:.*]] = insertvalue { i32, i32 } %[[MUL_A_B]], i32 %[[C_IMAG]], 1
// LLVM-INT: store { i32, i32 } %[[RESULT]], ptr %[[C_ADDR]], align 4

// OGCG-INT: %[[A_ADDR:.*]] = alloca { i32, i32 }, align 4
// OGCG-INT: %[[B_ADDR:.*]] = alloca { i32, i32 }, align 4
// OGCG-INT: %[[C_ADDR:.*]] = alloca { i32, i32 }, align 4
// OGCG-INT: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG-INT: %[[A_REAL:.*]] = load i32, ptr %[[A_REAL_PTR]], align 4
// OGCG-INT: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG-INT: %[[A_IMAG:.*]] = load i32, ptr %[[A_IMAG_PTR]], align 4
// OGCG-INT: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG-INT: %[[B_REAL:.*]] = load i32, ptr %[[B_REAL_PTR]], align 4
// OGCG-INT: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG-INT: %[[B_IMAG:.*]] = load i32, ptr %[[B_IMAG_PTR]], align 4
// OGCG-INT: %[[MUL_AR_BR:.*]] = mul i32 %[[A_REAL]], %[[B_REAL]]
// OGCG-INT: %[[MUL_AI_BI:.*]] = mul i32 %[[A_IMAG]], %[[B_IMAG]]
// OGCG-INT: %[[C_REAL:.*]] = sub i32 %[[MUL_AR_BR]], %[[MUL_AI_BI]]
// OGCG-INT: %[[MUL_AI_BR:.*]] = mul i32 %[[A_IMAG]], %[[B_REAL]]
// OGCG-INT: %[[MUL_AR_BI:.*]] = mul i32 %[[A_REAL]], %[[B_IMAG]]
// OGCG-INT: %[[C_IMAG:.*]] = add i32 %[[MUL_AI_BR]], %[[MUL_AR_BI]]
// OGCG-INT: %[[C_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[C_ADDR]], i32 0, i32 0
// OGCG-INT: %[[C_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[C_ADDR]], i32 0, i32 1
// OGCG-INT: store i32 %[[C_REAL]], ptr %[[C_REAL_PTR]], align 4
// OGCG-INT: store i32 %[[C_IMAG]], ptr %[[C_IMAG_PTR]], align 4

void foo2() {
  float _Complex a;
  float b;
  float _Complex c = a * b;
}

// CIR-COMBINED: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR-COMBINED: %[[B_ADDR:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["b"]
// CIR-COMBINED: %[[C_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["c", init]
// CIR-COMBINED: %[[TMP_A:.*]] = cir.load{{.*}} %0 : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR-COMBINED: %[[TMP_B:.*]] = cir.load{{.*}} %1 : !cir.ptr<!cir.float>, !cir.float
// CIR-COMBINED: %[[A_REAL:.*]] = cir.complex.real %[[TMP_A]] : !cir.complex<!cir.float> -> !cir.float
// CIR-COMBINED: %[[A_IMAG:.*]] = cir.complex.imag %[[TMP_A]] : !cir.complex<!cir.float> -> !cir.float
// CIR-COMBINED: %[[RESULT_REAL:.*]] = cir.binop(mul, %[[A_REAL]], %[[TMP_B]]) : !cir.float
// CIR-COMBINED: %[[RESULT_IMAG:.*]] = cir.binop(mul, %[[A_IMAG]], %[[TMP_B]]) : !cir.float
// CIR-COMBINED: %[[RESULT:.*]] = cir.complex.create %[[RESULT_REAL]], %[[RESULT_IMAG]] : !cir.float -> !cir.complex<!cir.float>
// CIR-COMBINED: cir.store{{.*}} %[[RESULT]], %[[C_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// LLVM-COMBINED: %[[A_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM-COMBINED: %[[B_ADDR:.*]] = alloca float, i64 1, align 4
// LLVM-COMBINED: %[[C_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM-COMBINED: %[[TMP_A:.*]] = load { float, float }, ptr %[[A_ADDR]], align 4
// LLVM-COMBINED: %[[TMP_B:.*]] = load float, ptr %[[B_ADDR]], align 4
// LLVM-COMBINED: %[[A_REAL:.*]] = extractvalue { float, float } %[[TMP_A]], 0
// LLVM-COMBINED: %[[A_IMAG:.*]] = extractvalue { float, float } %[[TMP_A]], 1
// LLVM-COMBINED: %[[RESULT_REAL:.*]] = fmul float %[[A_REAL]], %[[TMP_B]]
// LLVM-COMBINED: %[[RESULT_IMAG:.*]] = fmul float %[[A_IMAG]], %[[TMP_B]]
// LLVM-COMBINED: %[[TMP_RESULT:.*]] = insertvalue { float, float } {{.*}}, float %[[RESULT_REAL]], 0
// LLVM-COMBINED: %[[RESULT:.*]] = insertvalue { float, float } %[[TMP_RESULT]], float %[[RESULT_IMAG]], 1
// LLVM-COMBINED: store { float, float } %[[RESULT]], ptr %[[C_ADDR]], align 4

// OGCG-COMBINED: %[[A_ADDR:.*]] = alloca { float, float }, align 4
// OGCG-COMBINED: %[[B_ADDR:.*]] = alloca float, align 4
// OGCG-COMBINED: %[[C_ADDR:.*]] = alloca { float, float }, align 4
// OGCG-COMBINED: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG-COMBINED: %[[A_REAL:.*]] = load float, ptr %[[A_REAL_PTR]], align 4
// OGCG-COMBINED: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG-COMBINED: %[[A_IMAG:.*]] = load float, ptr %[[A_IMAG_PTR]], align 4
// OGCG-COMBINED: %[[TMP_B:.*]] = load float, ptr %[[B_ADDR]], align 4
// OGCG-COMBINED: %[[RESULT_REAL:.*]] = fmul float %[[A_REAL]], %[[TMP_B]]
// OGCG-COMBINED: %[[RESULT_IMAG:.*]] = fmul float %[[A_IMAG]], %[[TMP_B]]
// OGCG-COMBINED: %[[C_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[C_ADDR]], i32 0, i32 0
// OGCG-COMBINED: %[[C_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[C_ADDR]], i32 0, i32 1
// OGCG-COMBINED: store float %[[RESULT_REAL]], ptr %[[C_REAL_PTR]], align 4
// OGCG-COMBINED: store float %[[RESULT_IMAG]], ptr %[[C_IMAG_PTR]], align 4
