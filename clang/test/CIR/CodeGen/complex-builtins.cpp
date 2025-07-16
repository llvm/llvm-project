// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

void foo() {
 int _Complex a;
 int _Complex b;
 int _Complex r = __builtin_choose_expr(true, a, b);
}

// CIR: %[[COMPLEX_A:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["a"]
// CIR: %[[COMPLEX_R:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["r", init]
// CIR: %[[TMP_A:.*]] = cir.load{{.*}} %[[COMPLEX_A]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR: cir.store{{.*}} %[[TMP_A]], %[[COMPLEX_R]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>

// LLVM: %[[COMPLEX_A:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[COMPLEX_B:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[COMPLEX_R:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[TMP_A:.*]] = load { i32, i32 }, ptr %[[COMPLEX_A]], align 4
// LLVM: store { i32, i32 } %[[TMP_A]], ptr %[[COMPLEX_R]], align 4

// OGCG: %[[COMPLEX_A:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[COMPLEX_B:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[COMPLEX_R:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[COMPLEX_A]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load i32, ptr %[[A_REAL_PTR]], align 4
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[COMPLEX_A]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load i32, ptr %[[A_IMAG_PTR]], align 4
// OGCG: %[[R_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[COMPLEX_R]], i32 0, i32 0
// OGCG: %[[R_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[COMPLEX_R]], i32 0, i32 1
// OGCG: store i32 %[[A_REAL]], ptr %[[R_REAL_PTR]], align 4
// OGCG: store i32 %[[A_IMAG]], ptr %[[R_IMAG_PTR]], align 4

void foo2() {
  double _Complex a;
  double real = __builtin_creal(a);
}

// CIR: %[[COMPLEX:.*]] = cir.alloca !cir.complex<!cir.double>, !cir.ptr<!cir.complex<!cir.double>>, ["a"]
// CIR: %[[INIT:.*]] = cir.alloca !cir.double, !cir.ptr<!cir.double>, ["real", init]
// CIR: %[[TMP:.*]] = cir.load{{.*}} %[[COMPLEX]] : !cir.ptr<!cir.complex<!cir.double>>, !cir.complex<!cir.double>
// CIR: %[[REAL:.*]] = cir.complex.real %[[TMP]] : !cir.complex<!cir.double> -> !cir.double
// CIR: cir.store{{.*}} %[[REAL]], %[[INIT]] : !cir.double, !cir.ptr<!cir.double>

// LLVM: %[[COMPLEX:.*]] = alloca { double, double }, i64 1, align 8
// LLVM: %[[INIT:.*]] = alloca double, i64 1, align 8
// LLVM: %[[TMP:.*]] = load { double, double }, ptr %[[COMPLEX]], align 8
// LLVM: %[[REAL:.*]] = extractvalue { double, double } %[[TMP]], 0
// LLVM: store double %[[REAL]], ptr %[[INIT]], align 8

// OGCG: %[[COMPLEX:.*]] = alloca { double, double }, align 8
// OGCG: %[[INIT:.*]] = alloca double, align 8
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { double, double }, ptr %[[COMPLEX]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load double, ptr %[[A_REAL_PTR]], align 8
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { double, double }, ptr %[[COMPLEX]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load double, ptr %[[A_IMAG_PTR]], align 8
// OGCG: store double %[[A_REAL]], ptr %[[INIT]], align 8

void foo3() {
  double _Complex a;
  double imag = __builtin_cimag(a);
}

// CIR: %[[COMPLEX:.*]] = cir.alloca !cir.complex<!cir.double>, !cir.ptr<!cir.complex<!cir.double>>, ["a"]
// CIR: %[[INIT:.*]] = cir.alloca !cir.double, !cir.ptr<!cir.double>, ["imag", init]
// CIR: %[[TMP:.*]] = cir.load{{.*}} %[[COMPLEX]] : !cir.ptr<!cir.complex<!cir.double>>, !cir.complex<!cir.double>
// CIR: %[[IMAG:.*]] = cir.complex.imag %[[TMP]] : !cir.complex<!cir.double> -> !cir.double
// CIR: cir.store{{.*}} %[[IMAG]], %[[INIT]] : !cir.double, !cir.ptr<!cir.double>

// LLVM: %[[COMPLEX:.*]] = alloca { double, double }, i64 1, align 8
// LLVM: %[[INIT:.*]] = alloca double, i64 1, align 8
// LLVM: %[[TMP:.*]] = load { double, double }, ptr %[[COMPLEX]], align 8
// LLVM: %[[IMAG:.*]] = extractvalue { double, double } %[[TMP]], 1
// LLVM: store double %[[IMAG]], ptr %[[INIT]], align 8

// OGCG: %[[COMPLEX:.*]] = alloca { double, double }, align 8
// OGCG: %[[INIT:.*]] = alloca double, align 8
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { double, double }, ptr %[[COMPLEX]], i32 0, i32 0
// OGCG: %[[A_REAL:.*]] = load double, ptr %[[A_REAL_PTR]], align 8
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { double, double }, ptr %[[COMPLEX]], i32 0, i32 1
// OGCG: %[[A_IMAG:.*]] = load double, ptr %[[A_IMAG_PTR]], align 8
// OGCG: store double %[[A_IMAG]], ptr %[[INIT]], align 8
