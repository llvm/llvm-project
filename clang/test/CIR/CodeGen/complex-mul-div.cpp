// complex-range basic
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -complex-range=basic -Wno-unused-value -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-canonicalize -o %t.cir %s 2>&1 | FileCheck --check-prefix=CIR-BEFORE-BASIC %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -complex-range=basic -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefixes=CIR-AFTER-INT,CIR-AFTER-MUL-COMBINED,CIR-COMBINED,CIR-AFTER-BASIC
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -complex-range=basic -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefixes=LLVM-INT,LLVM-MUL-COMBINED,LLVM-COMBINED,LLVM-BASIC
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -complex-range=basic -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefixes=OGCG-INT,OGCG-MUL-COMBINED,OGCG-COMBINED,OGCG-BASIC

// complex-range improved
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -complex-range=improved -Wno-unused-value -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-canonicalize -o %t.cir %s 2>&1 | FileCheck --check-prefix=CIR-BEFORE-IMPROVED %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -complex-range=improved -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefixes=CIR-AFTER-INT,CIR-AFTER-MUL-COMBINED,CIR-COMBINED,CIR-AFTER-IMPROVED
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -complex-range=improved -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefixes=LLVM-INT,LLVM-MUL-COMBINED,LLVM-COMBINED,LLVM-IMPROVED
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -complex-range=improved -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefixes=OGCG-INT,OGCG-MUL-COMBINED,OGCG-COMBINED,OGCG-IMPROVED

// complex-range promoted
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -complex-range=promoted -Wno-unused-value -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-canonicalize -o %t.cir %s 2>&1 | FileCheck --check-prefix=CIR-BEFORE-PROMOTED %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -complex-range=promoted -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefixes=CIR-AFTER-INT,CIR-AFTER-MUL-COMBINED,CIR-COMBINED,CIR-AFTER-PROMOTED
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -complex-range=promoted -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefixes=LLVM-INT,LLVM-MUL-COMBINED,LLVM-COMBINED,LLVM-PROMOTED
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -complex-range=promoted -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefixes=OGCG-INT,OGCG-MUL-COMBINED,OGCG-COMBINED,OGCG-PROMOTED

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

void foo3() {
  float _Complex a;
  float _Complex b;
  float _Complex c = a / b;
}

// CIR-BEFORE-BASIC: %{{.*}} = cir.complex.div {{.*}}, {{.*}} range(basic) : !cir.complex<!cir.float>

// CIR-AFTER-BASIC: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR-AFTER-BASIC: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["b"]
// CIR-AFTER-BASIC: %[[C_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["c", init]
// CIR-AFTER-BASIC: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR-AFTER-BASIC: %[[TMP_B:.*]] = cir.load{{.*}} %[[B_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR-AFTER-BASIC: %[[A_REAL:.*]] = cir.complex.real %[[TMP_A]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER-BASIC: %[[A_IMAG:.*]] = cir.complex.imag %[[TMP_A]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER-BASIC: %[[B_REAL:.*]] = cir.complex.real %[[TMP_B]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER-BASIC: %[[B_IMAG:.*]] = cir.complex.imag %[[TMP_B]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER-BASIC: %[[MUL_AR_BR:.*]] = cir.binop(mul, %[[A_REAL]], %[[B_REAL]]) : !cir.float
// CIR-AFTER-BASIC: %[[MUL_AI_BI:.*]] = cir.binop(mul, %[[A_IMAG]], %[[B_IMAG]]) : !cir.float
// CIR-AFTER-BASIC: %[[MUL_BR_BR:.*]] = cir.binop(mul, %[[B_REAL]], %[[B_REAL]]) : !cir.float
// CIR-AFTER-BASIC: %[[MUL_BI_BI:.*]] = cir.binop(mul, %[[B_IMAG]], %[[B_IMAG]]) : !cir.float
// CIR-AFTER-BASIC: %[[ADD_ARBR_AIBI:.*]] = cir.binop(add, %[[MUL_AR_BR]], %[[MUL_AI_BI]]) : !cir.float
// CIR-AFTER-BASIC: %[[ADD_BRBR_BIBI:.*]] = cir.binop(add, %[[MUL_BR_BR]], %[[MUL_BI_BI]]) : !cir.float
// CIR-AFTER-BASIC: %[[RESULT_REAL:.*]] = cir.binop(div, %[[ADD_ARBR_AIBI]], %[[ADD_BRBR_BIBI]]) : !cir.float
// CIR-AFTER-BASIC: %[[MUL_AI_BR:.*]] = cir.binop(mul, %[[A_IMAG]], %[[B_REAL]]) : !cir.float
// CIR-AFTER-BASIC: %[[MUL_AR_BI:.*]] = cir.binop(mul, %[[A_REAL]], %[[B_IMAG]]) : !cir.float
// CIR-AFTER-BASIC: %[[SUB_AIBR_ARBI:.*]] = cir.binop(sub, %[[MUL_AI_BR]], %[[MUL_AR_BI]]) : !cir.float
// CIR-AFTER-BASIC: %[[RESULT_IMAG:.*]] = cir.binop(div, %[[SUB_AIBR_ARBI]], %14) : !cir.float
// CIR-AFTER-BASIC: %[[RESULT:.*]] = cir.complex.create %[[RESULT_REAL]], %[[RESULT_IMAG]] : !cir.float -> !cir.complex<!cir.float>
// CIR-AFTER-BASIC: cir.store{{.*}} %[[RESULT]], %[[C_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// LLVM-BASIC: %[[A_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM-BASIC: %[[B_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM-BASIC: %[[C_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM-BASIC: %[[TMP_A:.*]] = load { float, float }, ptr %[[A_ADDR]], align 4
// LLVM-BASIC: %[[TMP_B:.*]] = load { float, float }, ptr %[[B_ADDR]], align 4
// LLVM-BASIC: %[[A_REAL:.*]] = extractvalue { float, float } %[[TMP_A]], 0
// LLVM-BASIC: %[[A_IMAG:.*]] = extractvalue { float, float } %[[TMP_A]], 1
// LLVM-BASIC: %[[B_REAL:.*]] = extractvalue { float, float } %[[TMP_B]], 0
// LLVM-BASIC: %[[B_IMAG:.*]] = extractvalue { float, float } %[[TMP_B]], 1
// LLVM-BASIC: %[[MUL_AR_BR:.*]] = fmul float %[[A_REAL]], %[[B_REAL]]
// LLVM-BASIC: %[[MUL_AI_BI:.*]] = fmul float %[[A_IMAG]], %[[B_IMAG]]
// LLVM-BASIC: %[[MUL_BR_BR:.*]] = fmul float %[[B_REAL]], %[[B_REAL]]
// LLVM-BASIC: %[[MUL_BI_BI:.*]] = fmul float %[[B_IMAG]], %[[B_IMAG]]
// LLVM-BASIC: %[[ADD_ARBR_AIBI:.*]] = fadd float %[[MUL_AR_BR]], %[[MUL_AI_BI]]
// LLVM-BASIC: %[[ADD_BRBR_BIBI:.*]] = fadd float %[[MUL_BR_BR]], %[[MUL_BI_BI]]
// LLVM-BASIC: %[[RESULT_REAL:.*]] = fdiv float %[[ADD_ARBR_AIBI]], %[[ADD_BRBR_BIBI]]
// LLVM-BASIC: %[[MUL_AI_BR:.*]] = fmul float %[[A_IMAG]], %[[B_REAL]]
// LLVM-BASIC: %[[MUL_BR_BI:.*]] = fmul float %[[A_REAL]], %[[B_IMAG]]
// LLVM-BASIC: %[[SUB_AIBR_BRBI:.*]] = fsub float %[[MUL_AI_BR]], %[[MUL_BR_BI]]
// LLVM-BASIC: %[[RESULT_IMAG:.*]] = fdiv float %[[SUB_AIBR_BRBI]], %[[ADD_BRBR_BIBI]]
// LLVM-BASIC: %[[TMP_RESULT:.*]] = insertvalue { float, float } {{.*}}, float %[[RESULT_REAL]], 0
// LLVM-BASIC: %[[RESULT:.*]] = insertvalue { float, float } %[[TMP_RESULT]], float %[[RESULT_IMAG]], 1
// LLVM-BASIC: store { float, float } %[[RESULT]], ptr %[[C_ADDR]], align 4

// OGCG-BASIC: %[[A_ADDR:.*]] = alloca { float, float }, align 4
// OGCG-BASIC: %[[B_ADDR:.*]] = alloca { float, float }, align 4
// OGCG-BASIC: %[[C_ADDR:.*]] = alloca { float, float }, align 4
// OGCG-BASIC: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG-BASIC: %[[A_REAL:.*]] = load float, ptr %[[A_REAL_PTR]], align 4
// OGCG-BASIC: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG-BASIC: %[[A_IMAG:.*]] = load float, ptr %[[A_IMAG_PTR]], align 4
// OGCG-BASIC: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG-BASIC: %[[B_REAL:.*]] = load float, ptr %[[B_REAL_PTR]], align 4
// OGCG-BASIC: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG-BASIC: %[[B_IMAG:.*]] = load float, ptr %[[B_IMAG_PTR]], align 4
// OGCG-BASIC: %[[MUL_AR_BR:.*]] = fmul float %[[A_REAL]], %[[B_REAL]]
// OGCG-BASIC: %[[MUL_AI_BI:.*]] = fmul float %[[A_IMAG]], %[[B_IMAG]]
// OGCG-BASIC: %[[ADD_ARBR_AIBI:.*]] = fadd float %[[MUL_AR_BR]], %[[MUL_AI_BI]]
// OGCG-BASIC: %[[MUL_BR_BR:.*]] = fmul float %[[B_REAL]], %[[B_REAL]]
// OGCG-BASIC: %[[MUL_BI_BI:.*]] = fmul float %[[B_IMAG]], %[[B_IMAG]]
// OGCG-BASIC: %[[ADD_BRBR_BIBI:.*]] = fadd float %[[MUL_BR_BR]], %[[MUL_BI_BI]]
// OGCG-BASIC: %[[MUL_AI_BR:.*]] = fmul float %[[A_IMAG]], %[[B_REAL]]
// OGCG-BASIC: %[[MUL_AR_BI:.*]] = fmul float %[[A_REAL]], %[[B_IMAG]]
// OGCG-BASIC: %[[SUB_AIBR_BRBI:.*]] = fsub float %[[MUL_AI_BR]], %[[MUL_AR_BI]]
// OGCG-BASIC: %[[RESULT_REAL:.*]] = fdiv float %[[ADD_ARBR_AIBI]], %[[ADD_BRBR_BIBI]]
// OGCG-BASIC: %[[RESULT_IMAG:.*]] = fdiv float %[[SUB_AIBR_BRBI]], %[[ADD_BRBR_BIBI]]
// OGCG-BASIC: %[[C_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[C_ADDR]], i32 0, i32 0
// OGCG-BASIC: %[[C_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[C_ADDR]], i32 0, i32 1
// OGCG-BASIC: store float %[[RESULT_REAL]], ptr %[[C_REAL_PTR]], align 4
// OGCG-BASIC: store float %[[RESULT_IMAG]], ptr %[[C_IMAG_PTR]], align 4

// CIR-BEFORE-IMPROVED: %{{.*}} = cir.complex.div {{.*}}, {{.*}} range(improved) : !cir.complex<!cir.float>

// CIR-AFTER-IMPROVED: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR-AFTER-IMPROVED: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["b"]
// CIR-AFTER-IMPROVED: %[[C_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["c", init]
// CIR-AFTER-IMPROVED: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR-AFTER-IMPROVED: %[[TMP_B:.*]] = cir.load{{.*}} %[[B_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR-AFTER-IMPROVED: %[[A_REAL:.*]] = cir.complex.real %[[TMP_A]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER-IMPROVED: %[[A_IMAG:.*]] = cir.complex.imag %[[TMP_A]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER-IMPROVED: %[[B_REAL:.*]] = cir.complex.real %[[TMP_B]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER-IMPROVED: %[[B_IMAG:.*]] = cir.complex.imag %[[TMP_B]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER-IMPROVED: %[[ABS_B_REAL:.*]] = cir.fabs %[[B_REAL]] : !cir.float
// CIR-AFTER-IMPROVED: %[[ABS_B_IMAG:.*]] = cir.fabs %[[B_IMAG]] : !cir.float
// CIR-AFTER-IMPROVED: %[[ABS_B_CMP:.*]] = cir.cmp(ge, %[[ABS_B_REAL]], %[[ABS_B_IMAG]]) : !cir.float, !cir.bool
// CIR-AFTER-IMPROVED: %[[RESULT:.*]] = cir.ternary(%[[ABS_B_CMP]], true {
// CIR-AFTER-IMPROVED:   %[[DIV_BI_BR:.*]] = cir.binop(div, %[[B_IMAG]], %[[B_REAL]]) : !cir.float
// CIR-AFTER-IMPROVED:   %[[MUL_DIV_BIBR_BI:.*]] = cir.binop(mul, %[[DIV_BI_BR]], %[[B_IMAG]]) : !cir.float
// CIR-AFTER-IMPROVED:   %[[ADD_BR_MUL_DIV_BIBR_BI:.*]] = cir.binop(add, %[[B_REAL]], %[[MUL_DIV_BIBR_BI]]) : !cir.float
// CIR-AFTER-IMPROVED:   %[[MUL_AI_DIV_BIBR:.*]] = cir.binop(mul, %[[A_IMAG]], %[[DIV_BI_BR]]) : !cir.float
// CIR-AFTER-IMPROVED:   %[[ADD_AR_MUL_AI_DIV_BIBR:.*]] = cir.binop(add, %[[A_REAL]], %[[MUL_AI_DIV_BIBR]]) : !cir.float
// CIR-AFTER-IMPROVED:   %[[RESULT_REAL:.*]] = cir.binop(div, %[[ADD_AR_MUL_AI_DIV_BIBR]], %[[ADD_BR_MUL_DIV_BIBR_BI]]) : !cir.float
// CIR-AFTER-IMPROVED:   %[[MUL_AR_DIV_BIBR:.*]] = cir.binop(mul, %[[A_REAL]], %[[DIV_BI_BR]]) : !cir.float
// CIR-AFTER-IMPROVED:   %[[SUB_AI_MUL_AR_DIV_BIBR:.*]] = cir.binop(sub, %[[A_IMAG]], %[[MUL_AR_DIV_BIBR]]) : !cir.float
// CIR-AFTER-IMPROVED:   %[[RESULT_IMAG:.*]] = cir.binop(div, %[[SUB_AI_MUL_AR_DIV_BIBR]], %[[ADD_BR_MUL_DIV_BIBR_BI]]) : !cir.float
// CIR-AFTER-IMPROVED:   %[[RESULT_COMPLEX:.*]] = cir.complex.create %[[RESULT_REAL]], %[[RESULT_IMAG]] : !cir.float -> !cir.complex<!cir.float>
// CIR-AFTER-IMPROVED:   cir.yield %[[RESULT_COMPLEX]] : !cir.complex<!cir.float>
// CIR-AFTER-IMPROVED: }, false {
// CIR-AFTER-IMPROVED:   %[[DIV_BR_BI:.*]] = cir.binop(div, %[[B_REAL]], %[[B_IMAG]]) : !cir.float
// CIR-AFTER-IMPROVED:   %[[MUL_DIV_BRBI_BR:.*]] = cir.binop(mul, %[[DIV_BR_BI]], %[[B_REAL]]) : !cir.float
// CIR-AFTER-IMPROVED:   %[[ADD_BI_MUL_DIV_BRBI_BR:.*]] = cir.binop(add, %[[B_IMAG]], %[[MUL_DIV_BRBI_BR]]) : !cir.float
// CIR-AFTER-IMPROVED:   %[[MUL_AR_DIV_BIBR:.*]] = cir.binop(mul, %[[A_REAL]], %[[DIV_BR_BI]]) : !cir.float
// CIR-AFTER-IMPROVED:   %[[ADD_MUL_AR_DIV_BRBI_AI:.*]] = cir.binop(add, %[[MUL_AR_DIV_BIBR]], %[[A_IMAG]]) : !cir.float
// CIR-AFTER-IMPROVED:   %[[RESULT_REAL:.*]] = cir.binop(div, %[[ADD_MUL_AR_DIV_BRBI_AI]], %[[ADD_BI_MUL_DIV_BRBI_BR]]) : !cir.float
// CIR-AFTER-IMPROVED:   %[[MUL_AI_DIV_BRBI:.*]] = cir.binop(mul, %[[A_IMAG]], %[[DIV_BR_BI]]) : !cir.float
// CIR-AFTER-IMPROVED:   %[[SUB_MUL_AI_DIV_BRBI_AR:.*]] = cir.binop(sub, %[[MUL_AI_DIV_BRBI]], %[[A_REAL]]) : !cir.float
// CIR-AFTER-IMPROVED:   %[[RESULT_IMAG:.*]] = cir.binop(div, %[[SUB_MUL_AI_DIV_BRBI_AR]], %[[ADD_BI_MUL_DIV_BRBI_BR]]) : !cir.float
// CIR-AFTER-IMPROVED:   %[[RESULT_COMPLEX:.*]] = cir.complex.create %[[RESULT_REAL]], %[[RESULT_IMAG]] : !cir.float -> !cir.complex<!cir.float>
// CIR-AFTER-IMPROVED:   cir.yield %[[RESULT_COMPLEX]] : !cir.complex<!cir.float>
// CIR-AFTER-IMPROVED: }) : (!cir.bool) -> !cir.complex<!cir.float>
// CIR-AFTER-IMPROVED: cir.store{{.*}} %[[RESULT]], %[[C_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// LLVM-IMPROVED: %[[A_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM-IMPROVED: %[[B_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM-IMPROVED: %[[C_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM-IMPROVED: %[[TMP_A:.*]] = load { float, float }, ptr %[[A_ADDR]], align 4
// LLVM-IMPROVED: %[[TMP_B:.*]] = load { float, float }, ptr %[[B_ADDR]], align 4
// LLVM-IMPROVED: %[[A_REAL:.*]] = extractvalue { float, float } %[[TMP_A]], 0
// LLVM-IMPROVED: %[[A_IMAG:.*]] = extractvalue { float, float } %[[TMP_A]], 1
// LLVM-IMPROVED: %[[B_REAL:.*]] = extractvalue { float, float } %[[TMP_B]], 0
// LLVM-IMPROVED: %[[B_IMAG:.*]] = extractvalue { float, float } %[[TMP_B]], 1
// LLVM-IMPROVED: %[[ABS_B_REAL:.*]] = call float @llvm.fabs.f32(float %[[B_REAL]])
// LLVM-IMPROVED: %[[ABS_B_IMAG:.*]] = call float @llvm.fabs.f32(float %[[B_IMAG]])
// LLVM-IMPROVED: %[[ABS_B_CMP:.*]] = fcmp oge float %[[ABS_B_REAL]], %[[ABS_B_IMAG]]
// LLVM-IMPROVED: br i1 %[[ABS_B_CMP]], label %[[ABS_BR_GT_ABS_BI:.*]], label %[[ABS_BR_LT_ABS_BI:.*]]
// LLVM-IMPROVED: [[ABS_BR_GT_ABS_BI]]:
// LLVM-IMPROVED:  %[[DIV_BI_BR:.*]] = fdiv float %[[B_IMAG]], %[[B_REAL]]
// LLVM-IMPROVED:  %[[MUL_DIV_BIBR_BI:.*]] = fmul float %[[DIV_BI_BR]], %[[B_IMAG]]
// LLVM-IMPROVED:  %[[ADD_BR_MUL_DIV_BIBR_BI:.*]] = fadd float %[[B_REAL]], %[[MUL_DIV_BIBR_BI]]
// LLVM-IMPROVED:  %[[MUL_AI_DIV_BIBR:.*]] = fmul float %[[A_IMAG]], %[[DIV_BI_BR]]
// LLVM-IMPROVED:  %[[ADD_AR_MUL_AI_DIV_BIBR:.*]] = fadd float %[[A_REAL]], %[[MUL_AI_DIV_BIBR]]
// LLVM-IMPROVED:  %[[RESULT_REAL:.*]] = fdiv float %[[ADD_AR_MUL_AI_DIV_BIBR]], %16
// LLVM-IMPROVED:  %[[MUL_AR_DIV_BIBR:.*]] = fmul float %[[A_REAL]], %[[DIV_BI_BR]]
// LLVM-IMPROVED:  %[[SUB_AI_MUL_AR_DIV_BIBR:.*]] = fsub float %[[A_IMAG]], %[[MUL_AR_DIV_BIBR]]
// LLVM-IMPROVED:  %[[RESULT_IMAG:.*]] = fdiv float %[[SUB_AI_MUL_AR_DIV_BIBR]], %[[ADD_BR_MUL_DIV_BIBR_BI]]
// LLVM-IMPROVED:  %[[TMP_THEN_RESULT:.*]] = insertvalue { float, float } {{.*}}, float %[[RESULT_REAL]], 0
// LLVM-IMPROVED:  %[[THEN_RESULT:.*]] = insertvalue { float, float } %[[TMP_THEN_RESULT]], float %[[RESULT_IMAG]], 1
// LLVM-IMPROVED:  br label %[[PHI_RESULT:.*]]
// LLVM-IMPROVED: [[ABS_BR_LT_ABS_BI]]:
// LLVM-IMPROVED:  %[[DIV_BR_BI:.*]] = fdiv float %[[B_REAL]], %[[B_IMAG]]
// LLVM-IMPROVED:  %[[MUL_DIV_BRBI_BR:.*]] = fmul float %[[DIV_BR_BI]], %[[B_REAL]]
// LLVM-IMPROVED:  %[[ADD_BI_MUL_DIV_BRBI_BR:.*]] = fadd float %[[B_IMAG]], %[[MUL_DIV_BRBI_BR]]
// LLVM-IMPROVED:  %[[MUL_AR_DIV_BRBI:.*]] = fmul float %[[A_REAL]], %[[DIV_BR_BI]]
// LLVM-IMPROVED:  %[[ADD_MUL_AR_DIV_BRBI_AI:.*]] = fadd float %[[MUL_AR_DIV_BRBI]], %[[A_IMAG]]
// LLVM-IMPROVED:  %[[RESULT_REAL:.*]] = fdiv float %[[ADD_MUL_AR_DIV_BRBI_AI]], %[[ADD_BI_MUL_DIV_BRBI_BR]]
// LLVM-IMPROVED:  %[[MUL_AI_DIV_BRBI:.*]] = fmul float %[[A_IMAG]], %[[DIV_BR_BI]]
// LLVM-IMPROVED:  %[[SUB_MUL_AI_DIV_BRBI_AR:.*]] = fsub float %[[MUL_AI_DIV_BRBI]], %[[A_REAL]]
// LLVM-IMPROVED:  %[[RESULT_IMAG:.*]] = fdiv float %[[SUB_MUL_AI_DIV_BRBI_AR]], %[[ADD_BI_MUL_DIV_BRBI_BR]]
// LLVM-IMPROVED:  %[[TMP_ELSE_RESULT:.*]] = insertvalue { float, float } {{.*}}, float %[[RESULT_REAL]], 0
// LLVM-IMPROVED:  %[[ELSE_RESULT:.*]] = insertvalue { float, float } %[[TMP_ELSE_RESULT]], float %[[RESULT_IMAG]], 1
// LLVM-IMPROVED:  br label %[[PHI_RESULT]]
// LLVM-IMPROVED: [[PHI_RESULT]]:
// LLVM-IMPROVED:  %[[RESULT:.*]] = phi { float, float } [ %[[ELSE_RESULT]], %[[ABS_BR_LT_ABS_BI]] ], [ %[[THEN_RESULT]], %[[ABS_BR_GT_ABS_BI]] ]
// LLVM-IMPROVED:  br label %[[STORE_RESULT:.*]]
// LLVM-IMPROVED: [[STORE_RESULT]]:
// LLVM-IMPROVED:  store { float, float } %[[RESULT]], ptr %[[C_ADDR]], align 4

// OGCG-IMPROVED: %[[A_ADDR:.*]] = alloca { float, float }, align 4
// OGCG-IMPROVED: %[[B_ADDR:.*]] = alloca { float, float }, align 4
// OGCG-IMPROVED: %[[C_ADDR:.*]] = alloca { float, float }, align 4
// OGCG-IMPROVED: %a.realp = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG-IMPROVED: %a.real = load float, ptr %a.realp, align 4
// OGCG-IMPROVED: %a.imagp = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG-IMPROVED: %a.imag = load float, ptr %a.imagp, align 4
// OGCG-IMPROVED: %b.realp = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG-IMPROVED: %b.real = load float, ptr %b.realp, align 4
// OGCG-IMPROVED: %b.imagp = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG-IMPROVED: %b.imag = load float, ptr %b.imagp, align 4
// OGCG-IMPROVED: %[[ABS_B_REAL:.*]] = call float @llvm.fabs.f32(float %[[B_REAL]])
// OGCG-IMPROVED: %[[ABS_B_IMAG:.*]] = call float @llvm.fabs.f32(float %[[B_IMAG]])
// OGCG-IMPROVED: %[[ABS_B_CMP:.*]] = fcmp ugt float %[[ABS_B_REAL]], %[[ABS_B_IMAG]]
// OGCG-IMPROVED: br i1 %[[ABS_B_CMP]], label %[[ABS_BR_GT_ABS_BI:.*]], label %[[ABS_BR_LT_ABS_BI:.*]]
// OGCG-IMPROVED: [[ABS_BR_GT_ABS_BI]]:
// OGCG-IMPROVED:  %[[DIV_BI_BR:.*]] = fdiv float %[[B_IMAG]], %[[B_REAL]]
// OGCG-IMPROVED:  %[[MUL_DIV_BIBR_BI:.*]] = fmul float %[[DIV_BI_BR]], %[[B_IMAG]]
// OGCG-IMPROVED:  %[[ADD_BR_MUL_DIV_BIBR_BI:.*]] = fadd float %[[B_REAL]], %[[MUL_DIV_BIBR_BI]]
// OGCG-IMPROVED:  %[[MUL_AI_DIV_BIBR:.*]] = fmul float %[[A_IMAG]], %[[DIV_BI_BR]]
// OGCG-IMPROVED:  %[[ADD_AR_MUL_AI_DIV_BIBR:.*]] = fadd float %[[A_REAL]], %[[MUL_AI_DIV_BIBR]]
// OGCG-IMPROVED:  %[[THEN_RESULT_REAL:.*]] = fdiv float %[[ADD_AR_MUL_AI_DIV_BIBR]], %[[ADD_BR_MUL_DIV_BIBR_BI]]
// OGCG-IMPROVED:  %[[MUL_AR_DIV_BI_BR:.*]] = fmul float %[[A_REAL]], %[[DIV_BI_BR]]
// OGCG-IMPROVED:  %[[SUB_AI_MUL_AR_DIV_BIBR:.*]] = fsub float %[[A_IMAG]], %[[MUL_AR_DIV_BI_BR]]
// OGCG-IMPROVED:  %[[THEN_RESULT_IMAG:.*]] = fdiv float %[[SUB_AI_MUL_AR_DIV_BIBR]], %[[ADD_BR_MUL_DIV_BIBR_BI]]
// OGCG-IMPROVED:  br label %[[STORE_RESULT:.*]]
// OGCG-IMPROVED: [[ABS_BR_LT_ABS_BI]]:
// OGCG-IMPROVED:  %[[DIV_BR_BI:.*]] = fdiv float %[[B_REAL]], %[[B_IMAG]]
// OGCG-IMPROVED:  %[[MUL_DIV_BRBI_BR:.*]] = fmul float %[[DIV_BR_BI]], %[[B_REAL]]
// OGCG-IMPROVED:  %[[ADD_BI_MUL_DIV_BRBI_BR:.*]] = fadd float %[[B_IMAG]], %[[MUL_DIV_BRBI_BR]]
// OGCG-IMPROVED:  %[[MUL_AR_DIV_BRBI:.*]] = fmul float %[[A_REAL]], %[[DIV_BR_BI]]
// OGCG-IMPROVED:  %[[ADD_MUL_AR_DIV_BRBI_AI:.*]] = fadd float %[[MUL_AR_DIV_BRBI]], %[[A_IMAG]]
// OGCG-IMPROVED:  %[[ELSE_RESULT_REAL:.*]] = fdiv float %[[ADD_MUL_AR_DIV_BRBI_AI]], %[[ADD_BI_MUL_DIV_BRBI_BR]]
// OGCG-IMPROVED:  %[[MUL_AI_DIV_BRBI:.*]] = fmul float %[[A_IMAG]], %[[DIV_BR_BI]]
// OGCG-IMPROVED:  %[[SUB_MUL_AI_DIV_BRBI_AR:.*]] = fsub float %[[MUL_AI_DIV_BRBI]], %[[A_REAL]]
// OGCG-IMPROVED:  %[[ELSE_RESULT_IMAG:.*]] = fdiv float %[[SUB_MUL_AI_DIV_BRBI_AR]], %[[ADD_BI_MUL_DIV_BRBI_BR]]
// OGCG-IMPROVED:  br label %[[STORE_RESULT]]
// OGCG-IMPROVED: [[STORE_RESULT]]:
// OGCG-IMPROVED:  %[[RESULT_REAL:.*]] = phi float [ %[[THEN_RESULT_REAL]], %[[ABS_BR_GT_ABS_BI]] ], [ %[[ELSE_RESULT_REAL]], %[[ABS_BR_LT_ABS_BI]] ]
// OGCG-IMPROVED:  %[[RESULT_IMAG:.*]] = phi float [ %[[THEN_RESULT_IMAG]], %[[ABS_BR_GT_ABS_BI]] ], [ %[[ELSE_RESULT_IMAG]], %[[ABS_BR_LT_ABS_BI]] ]
// OGCG-IMPROVED:  %[[C_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[C_ADDR]], i32 0, i32 0
// OGCG-IMPROVED:  %[[C_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[C_ADDR]], i32 0, i32 1
// OGCG-IMPROVED:  store float %[[RESULT_REAL]], ptr %[[C_REAL_PTR]], align 4
// OGCG-IMPROVED:  store float %[[RESULT_IMAG]], ptr %[[C_IMAG_PTR]], align 4

// CIR-BEFORE-PROMOTED: %{{.*}} = cir.complex.div {{.*}}, {{.*}} range(promoted) : !cir.complex<!cir.float>

// CIR-AFTER-PROMOTED: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR-AFTER-PROMOTED: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["b"]
// CIR-AFTER-PROMOTED: %[[C_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["c", init]
// CIR-AFTER-PROMOTED: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR-AFTER-PROMOTED: %[[TMP_B:.*]] = cir.load{{.*}} %[[B_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR-AFTER-PROMOTED: %[[A_REAL:.*]] = cir.complex.real %[[TMP_A]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER-PROMOTED: %[[A_IMAG:.*]] = cir.complex.imag %[[TMP_A]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER-PROMOTED: %[[B_REAL:.*]] = cir.complex.real %[[TMP_B]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER-PROMOTED: %[[B_IMAG:.*]] = cir.complex.imag %[[TMP_B]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER-PROMOTED: %[[A_REAL_F64:.*]] = cir.cast(floating, %[[A_REAL]] : !cir.float), !cir.double
// CIR-AFTER-PROMOTED: %[[A_IMAG_F64:.*]] = cir.cast(floating, %[[A_IMAG]] : !cir.float), !cir.double
// CIR-AFTER-PROMOTED: %[[B_REAL_F64:.*]] = cir.cast(floating, %[[B_REAL]] : !cir.float), !cir.double
// CIR-AFTER-PROMOTED: %[[B_IMAG_F64:.*]] = cir.cast(floating, %[[B_IMAG]] : !cir.float), !cir.double
// CIR-AFTER-PROMOTED: %[[MUL_AR_BR:.*]] = cir.binop(mul, %[[A_REAL_F64]], %[[B_REAL_F64]]) : !cir.double
// CIR-AFTER-PROMOTED: %[[MUL_AI_BI:.*]] = cir.binop(mul, %[[A_IMAG_F64]], %[[B_IMAG_F64]]) : !cir.double
// CIR-AFTER-PROMOTED: %[[MUL_BR_BR:.*]] = cir.binop(mul, %[[B_REAL_F64]], %[[B_REAL_F64]]) : !cir.double
// CIR-AFTER-PROMOTED: %[[MUL_BI_BI:.*]] = cir.binop(mul, %[[B_IMAG_F64]], %[[B_IMAG_F64]]) : !cir.double
// CIR-AFTER-PROMOTED: %[[ADD_ARBR_AIBI:.*]] = cir.binop(add, %[[MUL_AR_BR]], %[[MUL_AI_BI]]) : !cir.double
// CIR-AFTER-PROMOTED: %[[ADD_BRBR_BIBI:.*]] = cir.binop(add, %[[MUL_BR_BR]], %[[MUL_BI_BI]]) : !cir.double
// CIR-AFTER-PROMOTED: %[[RESULT_REAL:.*]] = cir.binop(div, %[[ADD_ARBR_AIBI]], %18) : !cir.double
// CIR-AFTER-PROMOTED: %[[MUL_AI_BR:.*]] = cir.binop(mul, %[[A_IMAG_F64]], %[[B_REAL_F64]]) : !cir.double
// CIR-AFTER-PROMOTED: %[[MUL_AR_BI:.*]] = cir.binop(mul, %[[A_REAL_F64]], %[[B_IMAG_F64]]) : !cir.double
// CIR-AFTER-PROMOTED: %[[SUB_AIBR_ARBI:.*]] = cir.binop(sub, %[[MUL_AI_BR]], %[[MUL_AR_BI]]) : !cir.double
// CIR-AFTER-PROMOTED: %[[RESULT_IMAG:.*]] = cir.binop(div, %[[SUB_AIBR_ARBI]], %[[ADD_BRBR_BIBI]]) : !cir.double
// CIR-AFTER-PROMOTED: %[[RESULT_F64:.*]] = cir.complex.create %[[RESULT_REAL]], %[[RESULT_IMAG]] : !cir.double -> !cir.complex<!cir.double>
// CIR-AFTER-PROMOTED: %[[RESULT_REAL_F64:.*]] = cir.complex.real %[[RESULT_F64]] : !cir.complex<!cir.double> -> !cir.double
// CIR-AFTER-PROMOTED: %[[RESULT_IMAG_F64:.*]] = cir.complex.imag %[[RESULT_F64]] : !cir.complex<!cir.double> -> !cir.double
// CIR-AFTER-PROMOTED: %[[RESULT_REAL_F32:.*]] = cir.cast(floating, %[[RESULT_REAL_F64]] : !cir.double), !cir.float
// CIR-AFTER-PROMOTED: %[[RESULT_IMAG_F32:.*]] = cir.cast(floating, %[[RESULT_IMAG_F64]] : !cir.double), !cir.float
// CIR-AFTER-PROMOTED: %[[RESULT_F32:.*]] = cir.complex.create %[[RESULT_REAL_F32]], %[[RESULT_IMAG_F32]] : !cir.float -> !cir.complex<!cir.float>
// CIR-AFTER-PROMOTED: cir.store{{.*}} %[[RESULT_F32]], %[[C_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// LLVM-PROMOTED: %[[A_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM-PROMOTED: %[[B_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM-PROMOTED: %[[C_ADDR:.*]] = alloca { float, float }, i64 1, align 4
// LLVM-PROMOTED: %[[TMP_A:.*]] = load { float, float }, ptr %[[A_ADDR]], align 4
// LLVM-PROMOTED: %[[TMP_B:.*]] = load { float, float }, ptr %[[B_ADDR]], align 4
// LLVM-PROMOTED: %[[A_REAL:.*]] = extractvalue { float, float } %[[TMP_A]], 0
// LLVM-PROMOTED: %[[A_IMAG:.*]] = extractvalue { float, float } %[[TMP_A]], 1
// LLVM-PROMOTED: %[[B_REAL:.*]] = extractvalue { float, float } %[[TMP_B]], 0
// LLVM-PROMOTED: %[[B_IMAG:.*]] = extractvalue { float, float } %[[TMP_B]], 1
// LLVM-PROMOTED: %[[A_REAL_F64:.*]] = fpext float %[[A_REAL]] to double
// LLVM-PROMOTED: %[[A_IMAG_F64:.*]] = fpext float %[[A_IMAG]] to double
// LLVM-PROMOTED: %[[B_REAL_F64:.*]] = fpext float %[[B_REAL]] to double
// LLVM-PROMOTED: %[[B_IMAG_F64:.*]] = fpext float %[[B_IMAG]] to double
// LLVM-PROMOTED: %[[MUL_AR_BR:.*]] = fmul double %[[A_REAL_F64]], %[[B_REAL_F64]]
// LLVM-PROMOTED: %[[MUL_AI_BI:.*]] = fmul double %[[A_IMAG_F64]], %[[B_IMAG_F64]]
// LLVM-PROMOTED: %[[MUL_BR_BR:.*]] = fmul double %[[B_REAL_F64]], %[[B_REAL_F64]]
// LLVM-PROMOTED: %[[MUL_BI_BI:.*]] = fmul double %[[B_IMAG_F64]], %[[B_IMAG_F64]]
// LLVM-PROMOTED: %[[ADD_ARBR_AIBI:.*]] = fadd double %[[MUL_AR_BR]], %[[MUL_AI_BI]]
// LLVM-PROMOTED: %[[ADD_BRBR_BIBI:.*]] = fadd double %[[MUL_BR_BR]], %[[MUL_BI_BI]]
// LLVM-PROMOTED: %[[RESULT_REAL:.*]] = fdiv double %[[ADD_ARBR_AIBI]], %[[ADD_BRBR_BIBI]]
// LLVM-PROMOTED: %[[MUL_AI_BR:.*]] = fmul double %[[A_IMAG_F64]], %[[B_REAL_F64]]
// LLVM-PROMOTED: %[[MUL_AR_BR:.*]] = fmul double %[[A_REAL_F64]], %[[B_IMAG_F64]]
// LLVM-PROMOTED: %[[SUB_AIBR_ARBI:.*]] = fsub double %[[MUL_AI_BR]], %[[MUL_AR_BR]]
// LLVM-PROMOTED: %[[RESULT_IMAG:.*]] = fdiv double %[[SUB_AIBR_ARBI]], %[[ADD_BRBR_BIBI]]
// LLVM-PROMOTED: %[[TMP_RESULT_F64:.*]] = insertvalue { double, double } {{.*}}, double %[[RESULT_REAL]], 0
// LLVM-PROMOTED: %[[RESULT_F64:.*]] = insertvalue { double, double } %[[TMP_RESULT_F64]], double %[[RESULT_IMAG]], 1
// LLVM-PROMOTED: %[[RESULT_REAL_F32:.*]] = fptrunc double %[[RESULT_REAL]] to float
// LLVM-PROMOTED: %[[RESULT_IMAG_F32:.*]] = fptrunc double %[[RESULT_IMAG]] to float
// LLVM-PROMOTED: %[[TMP_RESULT_F32:.*]] = insertvalue { float, float } {{.*}}, float %[[RESULT_REAL_F32]], 0
// LLVM-PROMOTED: %[[RESULT_F32:.*]] = insertvalue { float, float } %[[TMP_RESULT_F32]], float %[[RESULT_IMAG_F32]], 1
// LLVM-PROMOTED: store { float, float } %[[RESULT_F32]], ptr %[[C_ADDR]], align 4

// OGCG-PROMOTED: %[[A_ADDR:.*]] = alloca { float, float }, align 4
// OGCG-PROMOTED: %[[B_ADDR:.*]] = alloca { float, float }, align 4
// OGCG-PROMOTED: %[[C_ADDR:.*]] = alloca { float, float }, align 4
// OGCG-PROMOTED: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG-PROMOTED: %[[A_REAL:.*]] = load float, ptr %[[A_REAL_PTR]], align 4
// OGCG-PROMOTED: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG-PROMOTED: %[[A_IMAG:.*]] = load float, ptr %[[A_IMAG_PTR]], align 4
// OGCG-PROMOTED: %[[A_REAL_F64:.*]] = fpext float %[[A_REAL]] to double
// OGCG-PROMOTED: %[[A_IMAG_F64:.*]] = fpext float %[[A_IMAG]] to double
// OGCG-PROMOTED: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG-PROMOTED: %[[B_REAL:.*]] = load float, ptr %[[B_REAL_PTR]], align 4
// OGCG-PROMOTED: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG-PROMOTED: %[[B_IMAG:.*]] = load float, ptr %[[B_IMAG_PTR]], align 4
// OGCG-PROMOTED: %[[B_REAL_F64:.*]] = fpext float %[[B_REAL]] to double
// OGCG-PROMOTED: %[[B_IMAG_F64:.*]] = fpext float %[[B_IMAG]] to double
// OGCG-PROMOTED: %[[MUL_AR_BR:.*]] = fmul double %[[A_REAL_F64]], %[[B_REAL_F64]]
// OGCG-PROMOTED: %[[MUL_AI_BI:.*]] = fmul double %[[A_IMAG_F64]], %[[B_IMAG_F64]]
// OGCG-PROMOTED: %[[ADD_ARBR_AIBI:.*]] = fadd double %[[MUL_AR_BR]], %[[MUL_AI_BI]]
// OGCG-PROMOTED: %[[MUL_BR_BR:.*]] = fmul double %[[B_REAL_F64]], %[[B_REAL_F64]]
// OGCG-PROMOTED: %[[MUL_BI_BI:.*]] = fmul double %[[B_IMAG_F64]], %[[B_IMAG_F64]]
// OGCG-PROMOTED: %[[ADD_BRBR_BIBI:.*]] = fadd double %[[MUL_BR_BR]], %[[MUL_BI_BI]]
// OGCG-PROMOTED: %[[MUL_AI_BR:.*]] = fmul double %[[A_IMAG_F64]], %[[B_REAL_F64]]
// OGCG-PROMOTED: %[[MUL_AR_BI:.*]] = fmul double %[[A_REAL_F64]], %[[B_IMAG_F64]]
// OGCG-PROMOTED: %[[SUB_AIBR_ARBI:.*]] = fsub double %[[MUL_AI_BR]], %[[MUL_AR_BI]]
// OGCG-PROMOTED: %[[RESULT_REAL:.*]] = fdiv double %[[ADD_ARBR_AIBI]], %[[ADD_BRBR_BIBI]]
// OGCG-PROMOTED: %[[RESULT_IMAG:.*]] = fdiv double %[[SUB_AIBR_ARBI]], %[[ADD_BRBR_BIBI]]
// OGCG-PROMOTED: %[[UNPROMOTION_RESULT_REAL:.*]] = fptrunc double %[[RESULT_REAL]] to float
// OGCG-PROMOTED: %[[UNPROMOTION_RESULT_IMAG:.*]] = fptrunc double %[[RESULT_IMAG]] to float
// OGCG-PROMOTED: %[[C_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[C_ADDR]], i32 0, i32 0
// OGCG-PROMOTED: %[[C_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[C_ADDR]], i32 0, i32 1
// OGCG-PROMOTED: store float %[[UNPROMOTION_RESULT_REAL]], ptr %[[C_REAL_PTR]], align 4
// OGCG-PROMOTED: store float %[[UNPROMOTION_RESULT_IMAG]], ptr %[[C_IMAG_PTR]], align 4

// CIR-BEFORE-FULL: %{{.*}} = cir.complex.div {{.*}}, {{.*}} range(full) : !cir.complex<!cir.float>

// CIR-AFTER-FULL: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["a"]
// CIR-AFTER-FULL: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["b"]
// CIR-AFTER-FULL: %[[C_ADDR:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["c", init]
// CIR-AFTER-FULL: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR-AFTER-FULL: %[[TMP_B:.*]] = cir.load{{.*}} %[[B_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR-AFTER-FULL: %[[A_REAL:.*]] = cir.complex.real %[[TMP_A]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER-FULL: %[[A_IMAG:.*]] = cir.complex.imag %[[TMP_A]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER-FULL: %[[B_REAL:.*]] = cir.complex.real %[[TMP_B]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER-FULL: %[[B_IMAG:.*]] = cir.complex.imag %[[TMP_B]] : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER-FULL: %[[RESULT:.*]] = cir.call @__divsc3(%[[A_REAL]], %[[A_IMAG]], %[[B_REAL]], %[[B_IMAG]]) : (!cir.float, !cir.float, !cir.float, !cir.float) -> !cir.complex<!cir.float>
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
// LLVM-FULL: %[[RESULT:.*]] = call { float, float } @__divsc3(float %[[A_REAL]], float %[[A_IMAG]], float %[[B_REAL]], float %[[B_IMAG]])
// LLVM-FULL: store { float, float } %[[RESULT]], ptr %[[C_ADDR]], align 4

// OGCG-FULL: %[[A_ADDR:.*]] = alloca { float, float }, align 4
// OGCG-FULL: %[[B_ADDR:.*]] = alloca { float, float }, align 4
// OGCG-FULL: %[[C_ADDR:.*]] = alloca { float, float }, align 4
// OGCG-FULL: %[[RESULT_ADDR:.*]] = alloca { float, float }, align 4
// OGCG-FULL: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG-FULL: %[[A_REAL:.*]] = load float, ptr %[[A_REAL_PTR]], align 4
// OGCG-FULL: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG-FULL: %[[A_IMAG:.*]] = load float, ptr %[[A_IMAG_PTR]], align 4
// OGCG-FULL: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG-FULL: %[[B_REAL:.*]] = load float, ptr %[[B_REAL_PTR]], align 4
// OGCG-FULL: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG-FULL: %[[B_IMAG:.*]] = load float, ptr %[[B_IMAG_PTR]], align 4
// OGCG-FULL: %[[RESULT:.*]] = call noundef <2 x float> @__divsc3(float noundef %[[A_REAL]], float noundef %[[A_IMAG]], float noundef %[[B_REAL]], float noundef %[[B_IMAG]]) #2
// OGCG-FULL: store <2 x float> %[[RESULT]], ptr %[[RESULT_ADDR]], align 4
// OGCG-FULL: %[[RESULT_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[RESULT_ADDR]], i32 0, i32 0
// OGCG-FULL: %[[RESULT_REAL:.*]] = load float, ptr %[[RESULT_REAL_PTR]], align 4
// OGCG-FULL: %[[RESULT_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[RESULT_ADDR]], i32 0, i32 1
// OGCG-FULL: %[[RESULT_IMAG:.*]] = load float, ptr %[[RESULT_IMAG_PTR]], align 4
// OGCG-FULL: %[[C_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[C_ADDR]], i32 0, i32 0
// OGCG-FULL: %[[C_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[C_ADDR]], i32 0, i32 1
// OGCG-FULL: store float %[[RESULT_REAL]], ptr %[[C_REAL_PTR]], align 4
// OGCG-FULL: store float %[[RESULT_IMAG]], ptr %[[C_IMAG_PTR]], align 4

void foo4() {
  int _Complex a;
  int _Complex b;
  int _Complex c = a / b;
}

// CIR-BEFORE-BASIC: %{{.*}} = cir.complex.div {{.*}}, {{.*}} range(basic) : !cir.complex<!s32i>

// CIR-BEFORE-IMPROVED: %{{.*}} = cir.complex.div {{.*}}, {{.*}} range(improved) : !cir.complex<!s32i>

// CIR-BEFORE-PROMOTED: %{{.*}} = cir.complex.div {{.*}}, {{.*}} range(promoted) : !cir.complex<!s32i>

// CIR-BEFORE-FULL: %{{.*}} = cir.complex.div {{.*}}, {{.*}} range(full) : !cir.complex<!s32i>

// CIR-COMBINED: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["a"]
// CIR-COMBINED: %[[B_ADDR:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["b"]
// CIR-COMBINED: %[[C_ADDR:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["c", init]
// CIR-COMBINED: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR-COMBINED: %[[TMP_B:.*]] = cir.load{{.*}} %[[B_ADDR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR-COMBINED: %[[A_REAL:.*]] = cir.complex.real %[[TMP_A]] : !cir.complex<!s32i> -> !s32i
// CIR-COMBINED: %[[A_IMAG:.*]] = cir.complex.imag %[[TMP_A]] : !cir.complex<!s32i> -> !s32i
// CIR-COMBINED: %[[B_REAL:.*]] = cir.complex.real %[[TMP_B]] : !cir.complex<!s32i> -> !s32i
// CIR-COMBINED: %[[B_IMAG:.*]] = cir.complex.imag %[[TMP_B]] : !cir.complex<!s32i> -> !s32i
// CIR-COMBINED: %[[MUL_AR_BR:.*]] = cir.binop(mul, %[[A_REAL]], %[[B_REAL]]) : !s32i
// CIR-COMBINED: %[[MUL_AI_BI:.*]] = cir.binop(mul, %[[A_IMAG]], %[[B_IMAG]]) : !s32i
// CIR-COMBINED: %[[MUL_BR_BR:.*]] = cir.binop(mul, %[[B_REAL]], %[[B_REAL]]) : !s32i
// CIR-COMBINED: %[[MUL_BI_BI:.*]] = cir.binop(mul, %[[B_IMAG]], %[[B_IMAG]]) : !s32i
// CIR-COMBINED: %[[ADD_ARBR_AIBI:.*]] = cir.binop(add, %[[MUL_AR_BR]], %[[MUL_AI_BI]]) : !s32i
// CIR-COMBINED: %[[ADD_BRBR_BIBI:.*]] = cir.binop(add, %[[MUL_BR_BR]], %[[MUL_BI_BI]]) : !s32i
// CIR-COMBINED: %[[RESULT_REAL:.*]] = cir.binop(div, %[[ADD_ARBR_AIBI]], %[[ADD_BRBR_BIBI]]) : !s32i
// CIR-COMBINED: %[[MUL_AI_BR:.*]] = cir.binop(mul, %[[A_IMAG]], %[[B_REAL]]) : !s32i
// CIR-COMBINED: %[[MUL_AR_BI:.*]] = cir.binop(mul, %[[A_REAL]], %[[B_IMAG]]) : !s32i
// CIR-COMBINED: %[[SUB_AIBR_ARBI:.*]] = cir.binop(sub, %[[MUL_AI_BR]], %[[MUL_AR_BI]]) : !s32i
// CIR-COMBINED: %[[RESULT_IMAG:.*]] = cir.binop(div, %[[SUB_AIBR_ARBI]], %14) : !s32i
// CIR-COMBINED: %[[RESULT:.*]] = cir.complex.create %[[RESULT_REAL]], %[[RESULT_IMAG]] : !s32i -> !cir.complex<!s32i>
// CIR-COMBINED: cir.store{{.*}} %[[RESULT]], %[[C_ADDR]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>

// LLVM-COMBINED: %[[A_ADDR:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM-COMBINED: %[[B_ADDR:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM-COMBINED: %[[C_ADDR:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM-COMBINED: %[[TMP_A:.*]] = load { i32, i32 }, ptr %[[A_ADDR]], align 4
// LLVM-COMBINED: %[[TMP_B:.*]] = load { i32, i32 }, ptr %[[B_ADDR]], align 4
// LLVM-COMBINED: %[[A_REAL:.*]] = extractvalue { i32, i32 } %[[TMP_A]], 0
// LLVM-COMBINED: %[[A_IMAG:.*]] = extractvalue { i32, i32 } %[[TMP_A]], 1
// LLVM-COMBINED: %[[B_REAL:.*]] = extractvalue { i32, i32 } %[[TMP_B]], 0
// LLVM-COMBINED: %[[B_IMAG:.*]] = extractvalue { i32, i32 } %[[TMP_B]], 1
// LLVM-COMBINED: %[[MUL_AR_BR:.*]] = mul i32 %[[A_REAL]], %[[B_REAL]]
// LLVM-COMBINED: %[[MUL_AI_BI:.*]] = mul i32 %[[A_IMAG]], %[[B_IMAG]]
// LLVM-COMBINED: %[[MUL_BR_BR:.*]] = mul i32 %[[B_REAL]], %[[B_REAL]]
// LLVM-COMBINED: %[[MUL_BI_BI:.*]] = mul i32 %[[B_IMAG]], %[[B_IMAG]]
// LLVM-COMBINED: %[[ADD_ARBR_AIBI:.*]] = add i32 %[[MUL_AR_BR]], %[[MUL_AI_BI]]
// LLVM-COMBINED: %[[ADD_BRBR_BIBI:.*]] = add i32 %[[MUL_BR_BR]], %[[MUL_BI_BI]]
// LLVM-COMBINED: %[[RESULT_REAL:.*]] = sdiv i32 %[[ADD_ARBR_AIBI]], %[[ADD_BRBR_BIBI]]
// LLVM-COMBINED: %[[MUL_AI_BR:.*]] = mul i32 %[[A_IMAG]], %[[B_REAL]]
// LLVM-COMBINED: %[[MUL_BR_BI:.*]] = mul i32 %[[A_REAL]], %[[B_IMAG]]
// LLVM-COMBINED: %[[SUB_AIBR_BRBI:.*]] = sub i32 %[[MUL_AI_BR]], %[[MUL_BR_BI]]
// LLVM-COMBINED: %[[RESULT_IMAG:.*]] = sdiv i32 %[[SUB_AIBR_BRBI]], %[[ADD_BRBR_BIBI]]
// LLVM-COMBINED: %[[TMP_RESULT:.*]] = insertvalue { i32, i32 } {{.*}}, i32 %[[RESULT_REAL]], 0
// LLVM-COMBINED: %[[RESULT:.*]] = insertvalue { i32, i32 } %[[TMP_RESULT]], i32 %[[RESULT_IMAG]], 1
// LLVM-COMBINED: store { i32, i32 } %[[RESULT]], ptr %[[C_ADDR]], align 4

// OGCG-COMBINED: %[[A_ADDR:.*]] = alloca { i32, i32 }, align 4
// OGCG-COMBINED: %[[B_ADDR:.*]] = alloca { i32, i32 }, align 4
// OGCG-COMBINED: %[[C_ADDR:.*]] = alloca { i32, i32 }, align 4
// OGCG-COMBINED: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG-COMBINED: %[[A_REAL:.*]] = load i32, ptr %[[A_REAL_PTR]], align 4
// OGCG-COMBINED: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG-COMBINED: %[[A_IMAG:.*]] = load i32, ptr %[[A_IMAG_PTR]], align 4
// OGCG-COMBINED: %[[B_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[B_ADDR]], i32 0, i32 0
// OGCG-COMBINED: %[[B_REAL:.*]] = load i32, ptr %[[B_REAL_PTR]], align 4
// OGCG-COMBINED: %[[B_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[B_ADDR]], i32 0, i32 1
// OGCG-COMBINED: %[[B_IMAG:.*]] = load i32, ptr %[[B_IMAG_PTR]], align 4
// OGCG-COMBINED: %[[MUL_AR_BR:.*]] = mul i32 %[[A_REAL]], %[[B_REAL]]
// OGCG-COMBINED: %[[MUL_AI_BI:.*]] = mul i32 %[[A_IMAG]], %[[B_IMAG]]
// OGCG-COMBINED: %[[ADD_ARBR_AIBI:.*]] = add i32 %[[MUL_AR_BR]], %[[MUL_AI_BI]]
// OGCG-COMBINED: %[[MUL_BR_BR:.*]] = mul i32 %[[B_REAL]], %[[B_REAL]]
// OGCG-COMBINED: %[[MUL_BI_BI:.*]] = mul i32 %[[B_IMAG]], %[[B_IMAG]]
// OGCG-COMBINED: %[[ADD_BRBR_BIBI:.*]] = add i32 %[[MUL_BR_BR]], %[[MUL_BI_BI]]
// OGCG-COMBINED: %[[MUL_AI_BR:.*]] = mul i32 %[[A_IMAG]], %[[B_REAL]]
// OGCG-COMBINED: %[[MUL_AR_BI:.*]] = mul i32 %[[A_REAL]], %[[B_IMAG]]
// OGCG-COMBINED: %[[SUB_AIBR_BRBI:.*]] = sub i32 %[[MUL_AI_BR]], %[[MUL_AR_BI]]
// OGCG-COMBINED: %[[RESULT_REAL:.*]] = sdiv i32 %[[ADD_ARBR_AIBI]], %[[ADD_BRBR_BIBI]]
// OGCG-COMBINED: %[[RESULT_IMAG:.*]] = sdiv i32 %[[SUB_AIBR_BRBI]], %[[ADD_BRBR_BIBI]]
// OGCG-COMBINED: %[[C_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[C_ADDR]], i32 0, i32 0
// OGCG-COMBINED: %[[C_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[C_ADDR]], i32 0, i32 1
// OGCG-COMBINED: store i32 %[[RESULT_REAL]], ptr %[[C_REAL_PTR]], align 4
// OGCG-COMBINED: store i32 %[[RESULT_IMAG]], ptr %[[C_IMAG_PTR]], align 4
